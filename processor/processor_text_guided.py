# processor/processor_text_guided.py
import logging
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval

def do_train_text_guided(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_func, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training text-guided stage')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # 文本相关 meter
    loss_text_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    # 训练循环
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        loss_text_meter.reset()
        
        evaluator.reset()
        scheduler.step()
        
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            # 获取数据
            img = batch['images'].to(device)
            target = batch['labels'].to(device)
            target_cam = batch['camids'].to(device)
            # text = batch['text_tokens'].to(device) # 如果有
            
            # 前向传播 (图像部分)
            score, feat, image_features = model(img, target, cam_label=target_cam)
            
            # 计算图像损失
            loss_img = loss_func(score, feat, target, target_cam)
            
            # --- Text-Guided 部分 (目前仅占位，因模型尚未完全集成文本流) ---
            # 理想情况下：
            # text_features = model.text_encoder(text)
            # loss_text = cross_modal_loss(image_features, text_features, target)
            loss_text = torch.tensor(0.0).to(device) 
            
            loss = loss_img + loss_text
            
            # 反向传播
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
                
            loss.backward()
            
            optimizer.step()
            if optimizer_center is not None:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            # 记录日志
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        
        # 保存模型
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), 
                       os.path.join(output_dir, f"text_guided_model_{epoch}.pth"))

        # 验证
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            logger.info("Rank-1: {:.1%}".format(cmc[0]))
            torch.cuda.empty_cache()

def do_train_text_guided_stage1(cfg, model, train_loader, optimizer, scheduler, local_rank):
    logger = logging.getLogger("transreid.train")
    logger.info("Enter Text-Guided Stage 1 Training")
    
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    
    loss_meter = AverageMeter()
    
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 简单的 Stage 1 循环 (对比学习预训练)
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        
    # Stage 1: Text-Image Contrastive Learning
    # 使用 SupConLoss (需要确保已定义或导入)
    from loss.supcontrast import SupConLoss
    xent = SupConLoss(device)
    
    # 启用 AMP
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            img = batch['images'].to(device)
            target = batch['labels'].to(device)
            
            # 如果 batch 包含 camids，我们也可以传进去
            camids = batch.get('camids')
            if camids is not None:
                camids = camids.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=True):
                # 1. 计算图像特征 (get_image=True)
                # 注意：make_model_clipreid 需要 x=img
                # 返回: image_features_last, image_features, image_features_proj
                # 这里我们假设 model.forward 逻辑没变，我们主要需要 projected features
                if cfg.MODEL.SIE_CAMERA and camids is not None:
                    img_out = model(x=img, label=target, get_image=True, cam_label=camids)
                else:
                    img_out = model(x=img, label=target, get_image=True)
                
                # ViT-B-16 返回的是 features[:, 0] (CLS token)
                # img_out 就是最终投影后的特征
                image_features = img_out
                
                # 2. 计算文本特征 (get_text=True)
                # 这会使用 prompt learner 生成 prompt 并编码
                text_features = model(label=target, get_text=True)
                
                # 3. 计算损失
                loss_i2t = xent(image_features, text_features, target, target)
                loss_t2i = xent(text_features, image_features, target, target)
                loss = loss_i2t + loss_t2i
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_meter.update(loss.item(), img.shape[0])
            
            if (n_iter + 1) % log_period == 0:
                logger.info("Stage1 Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

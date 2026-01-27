# processor/processor_text_guided.py
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def do_train_text_guided(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_func, num_query, local_rank):
    """Stage 2: 分类训练"""
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('Start Stage 2: Text-Guided Classification Training')
    
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            img = batch['images'].to(device)
            target = batch['labels'].to(device)
            target_cam = batch['camids'].to(device) if batch.get('camids') is not None else None
            
            # 前向传播
            if target_cam is not None and cfg.MODEL.SIE_CAMERA:
                score, feat, image_features = model(img, target, cam_label=target_cam)
            else:
                score, feat, image_features = model(img, target)
            
            # 计算损失
            i2t_logits = None
            text_tokens = batch.get('text_tokens')
            if text_tokens is not None:
                text_tokens = text_tokens.to(device)
                text_features = model(get_text_tokens=True, text_tokens=text_tokens)
                i2t_logits = image_features @ text_features.t()
            loss = loss_func(score, feat, target, target_cam, i2t_logits)
            
            # 反向传播
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
                
            loss.backward()
            
            optimizer.step()
            if optimizer_center is not None and center_criterion is not None:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / cfg.SOLVER.STAGE2.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            # 记录日志
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if (n_iter + 1) % log_period == 0:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, current_lr))

        end_time = time.time()
        logger.info("Epoch {} done. Time: {:.1f}s".format(epoch, end_time - start_time))
        
        # 保存模型
        if epoch % checkpoint_period == 0 or epoch == epochs:
            save_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_stage2_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved stage2 checkpoint to {save_path}")

        # 验证
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
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
    """Stage 1: 对比学习预训练"""
    logger = logging.getLogger("transreid.train")
    logger.info("Enter Text-Guided Stage 1 Training")
    
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    
    loss_meter = AverageMeter()
    
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Stage 1: Text-Image Contrastive Learning
    from loss.supcontrast import SupConLoss
    from loss.cross_modal_loss import CrossModalContrastiveLoss
    xent = SupConLoss(device)
    cm_contrast = CrossModalContrastiveLoss()
    # AMP 在此处容易触发 GradScaler 的 inf 检测异常，优先稳定训练
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            img = batch['images'].to(device)
            target = batch['labels'].to(device)
            camids = batch.get('camids')
            if camids is not None: 
                camids = camids.to(device)
            text_tokens = batch.get('text_tokens')
            if text_tokens is not None:
                text_tokens = text_tokens.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                if cfg.MODEL.SIE_CAMERA and camids is not None:
                    img_out = model(x=img, label=target, get_image=True, cam_label=camids)
                else:
                    img_out = model(x=img, label=target, get_image=True)

                image_features = img_out
                if text_tokens is not None:
                    text_features = model(get_text_tokens=True, text_tokens=text_tokens)
                    loss = cm_contrast(text_features, image_features, target)
                else:
                    text_features = model(label=target, get_text=True)
                    loss_i2t = xent(image_features, text_features, target, target)
                    loss_t2i = xent(text_features, image_features, target, target)
                    loss = loss_i2t + loss_t2i
            
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), img.shape[0])
            
            if (n_iter + 1) % log_period == 0:
                logger.info("Stage1 Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))
        
        # 保存模型
        if epoch % checkpoint_period == 0 or epoch == epochs:
            save_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_stage1_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved stage1 checkpoint to {save_path}")

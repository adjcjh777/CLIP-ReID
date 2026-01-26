
from utils.logger import setup_logger
from datasets.text_reid_dataset import make_text_dataloader
from model.make_model_clipreid import make_model
from model.text_query_encoder import build_text_encoder
# Use original optimizers and schedulers
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_text_guided import do_train_text_guided, do_train_text_guided_stage1
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text-Guided ReID Training")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--annotation_file", default="annotations/market1501_train.json", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Create Text-Guided DataLoader
    # Note: make_text_dataloader returns (loader, num_classes)
    # But do_train_stage2 needs val_loader. For now we use the original make_dataloader for val.
    from datasets.make_dataloader_clipreid import make_dataloader
    _, _, val_loader, num_query, num_classes_orig, cam_num, view_num = make_dataloader(cfg)
    
    # Text Dataloader
    # Note: Using CLIP tokenizer from clip library if needed, passed as none for now
    train_loader_text, num_classes_text = make_text_dataloader(cfg, args.annotation_file)
    
    num_classes = max(num_classes_orig, num_classes_text)

    # Model
    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    
    # Text Encoder (if we want to use it separately, but it's better integrated into the model)
    # For this quick prototype, we assume the model handles visual part, and we might add text encoder part
    # But for now, let's just train the visual part with the NEW dataloader to prove pipeline works
    
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    
    # Optimizers
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS, 
                                        lr_min=cfg.SOLVER.STAGE1.LR_MIN, 
                                        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT, 
                                        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS)

    # Stage 1
    do_train_text_guided_stage1(cfg, model, train_loader_text, optimizer_1stage, scheduler_1stage, args.local_rank)
    
    # Stage 2
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA,
                                         cfg.SOLVER.STAGE2.WARMUP_FACTOR, cfg.SOLVER.STAGE2.WARMUP_ITERS,
                                         cfg.SOLVER.STAGE2.WARMUP_METHOD)
                                         
    do_train_text_guided(cfg, model, center_criterion, train_loader_text, val_loader, 
                         optimizer_2stage, optimizer_center_2stage, scheduler_2stage, loss_func, 
                         num_query, args.local_rank)

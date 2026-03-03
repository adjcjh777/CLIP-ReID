#!/usr/bin/env python
"""
轻量 eval-only 脚本：对已有 checkpoint 做推理评估，支持 re-ranking。

用法：
  python scripts/eval_checkpoint.py \
    --config_file configs/person/vit_clipreid.yml \
    --checkpoint /path/to/ViT-B-16_stage2_60.pth \
    [--reranking] \
    [DATASETS.NAMES market1501] \
    [DATASETS.ROOT_DIR ./DATASETS]
"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.metrics import R1_mAP_eval
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("eval")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reranking", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.reranking:
        cfg.defrost()
        cfg.TEST.RE_RANKING = True
        cfg.freeze()
    cfg.freeze() if not cfg.is_frozen() else None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Dataset : {cfg.DATASETS.NAMES}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Re-ranking: {cfg.TEST.RE_RANKING}")

    # 数据加载
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    
    # 模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    evaluator = R1_mAP_eval(num_query, max_rank=50,
                             feat_norm=cfg.TEST.FEAT_NORM,
                             reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    logger.info("Extracting features...")
    with torch.no_grad():
        for img, vid, camid, camids, target_view, _ in val_loader:
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info(f"mAP  : {mAP:.1%}")
    logger.info(f"Rank-1: {cmc[0]:.1%}")
    logger.info(f"Rank-5: {cmc[4]:.1%}")
    logger.info(f"Rank-10: {cmc[9]:.1%}")


if __name__ == "__main__":
    main()

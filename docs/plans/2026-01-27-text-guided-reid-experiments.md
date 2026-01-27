# Text-Guided ReID 可复现实验计划（2026-01-27）

## 目标
- 在 **相同测试配置** 下，对比 Baseline / Baseline+SIE(+OLP) / Text-guided ReID。
- 找到 text-guided 低于 baseline 的主要原因，并验证哪些改动能缩小差距。

## 成功判据
- 统一测试配置（`INPUT.SIZE_{TRAIN,TEST}=384x128`, `TEST.IMS_PER_BATCH=256`）下得到可复现的 mAP/Rank-1/5/10。
- 对比表中每条实验都有：模型权重、日志路径、commit hash。

## 环境记录（务必写进日志/表格）
- GPU: RTX 4090
- CUDA / 驱动: 记录 `nvidia-smi`
- PyTorch / torchvision: 记录 `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`
- 仓库 commit：`git rev-parse HEAD`

## 数据与注释
- Market1501：`/root/autodl-tmp/CLIP_REID/DATASETS/Market-1501-v15.09.15`
- MSMT17：`/root/autodl-tmp/CLIP_REID/DATASETS/MSMT17`
- Text annotations:
  - Market1501: `annotations/market1501_train.json`
  - MSMT17: `annotations/msmt17_train.json`
- 属性文件：`datasets/attributes/market1501_attribute.mat`

## 统一测试配置（强制一致）
- `INPUT.SIZE_TRAIN=[384,128]`
- `INPUT.SIZE_TEST=[384,128]`
- `TEST.IMS_PER_BATCH=256`
- 评测入口：`test_clipreid.py`

## 实验矩阵（建议按顺序）

### A. Baseline 系列（可选重新训练）
1) **Baseline (CLIP-ReID)**
   - SIE: off, OLP: off
2) **Baseline + SIE**
   - `MODEL.SIE_CAMERA=True`（如需 SIE_VIEW 同理）
3) **Baseline + SIE + OLP**
   - 仅在已有 OLP 实现的分支运行

### B. Text-guided 系列
4) **Text-guided（当前实现）**
   - I2T loss 开启，SIE/OLP 关闭
5) **Text-guided + SIE**
   - `MODEL.SIE_CAMERA=True`，其余不变
6) **Text-guided + SIE + OLP**
   - 仅在 OLP 分支运行
7) **I2T 权重消融**
   - `MODEL.I2T_LOSS_WEIGHT`: 1.0 / 0.5 / 0.2
8) **文本来源消融**
   - attribute vs BLIP

## 训练与评测命令（模板）

### Baseline 训练（CLIP-ReID）
```
cd /root/CLIP-ReID
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR /root/autodl-tmp/CLIP_REID/DATASETS \
  OUTPUT_DIR /root/autodl-tmp/CLIP_REID/OUTPUT/vit_clipreid_market1501 \
  INPUT.SIZE_TRAIN "[384, 128]" \
  INPUT.SIZE_TEST "[384, 128]" \
  SOLVER.STAGE1.IMS_PER_BATCH 32 \
  SOLVER.STAGE2.IMS_PER_BATCH 32
```

### Text-guided 训练（Stage1+Stage2）
```
cd /root/CLIP-ReID
bash scripts/train_text_guided_pipeline.sh market1501 0 attribute
```
如需 SIE/OLP 或 I2T 权重消融，用命令行覆盖：
```
CUDA_VISIBLE_DEVICES=0 python train_text_reid.py \
  --config_file configs/person/vit_clipreid.yml \
  --annotation_file annotations/market1501_train.json \
  DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR /root/autodl-tmp/CLIP_REID/DATASETS \
  OUTPUT_DIR /root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/market1501 \
  MODEL.SIE_CAMERA True \
  MODEL.I2T_LOSS_WEIGHT 0.5
```

### 统一测试（Baseline/Text-guided 都用）
```
cd /root/CLIP-ReID
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  TEST.WEIGHT /path/to/ckpt.pth \
  DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR /root/autodl-tmp/CLIP_REID/DATASETS \
  OUTPUT_DIR /root/autodl-tmp/CLIP_REID/OUTPUT/your_exp_dir \
  INPUT.SIZE_TRAIN "[384, 128]" \
  INPUT.SIZE_TEST "[384, 128]" \
  TEST.IMS_PER_BATCH 256
```

## 结果记录模板（每次填）

| 实验 | 数据集 | Train/Test 尺寸 | SIE | OLP | I2T 权重 | 权重文件 | mAP | Rank-1 | Rank-5 | Rank-10 | 日志 | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | Market1501 | 384x128 | off | off | - | ... | ... | ... | ... | ... | ... | ... |
| Text-guided | Market1501 | 384x128 | off | off | 1.0 | ... | ... | ... | ... | ... | ... | ... |

## 风险与注意事项
- **位置编码维度不匹配**：当测试分辨率与训练不同，需确保可自适应 resize。
- **公平性**：对比必须统一测试分辨率、batch size、数据集与评测脚本。
- **随机性**：建议至少 3 个 seed（1234/777/4321）并报告均值与方差。

## 输出物
- 评测日志（包含 mAP/Rank-1/5/10）
- 对比表（Markdown 或 CSV）
- 关键配置参数与 commit hash

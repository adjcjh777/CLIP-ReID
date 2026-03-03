# 2026-03-02 本机训练计划（基于 full-progress-report）

## 目标

基于 `docs/plans/2026-03-02-full-progress-report.md`，今天在本机优先执行一条“高收益、低风险”的主线：

1. 复现 B5 最优基础配置（`I2T=0.2 + SIE + OLP`）
2. 训练结束后直接做 re-ranking 评估（E5）

这样可以对齐报告中已经验证过的有效方向，避免重复投入到已证实收益低或负收益的配置（如 Scheduled I2T）。

## 计划内容

### P0（立即执行）

- 实验名：`B5_local_rebuild`
- 数据集：`Market-1501`
- 配置：
  - `MODEL.TEXT_ENCODER_TYPE=clip_native`
  - `MODEL.TEXT_LOSS_TYPE=none`
  - `MODEL.I2T_LOSS_WEIGHT=0.2`
  - `MODEL.SIE_CAMERA=True`
  - `MODEL.SIE_COE=3.0`
  - `MODEL.STRIDE_SIZE=[12,12]`
  - `SOLVER.STAGE1.IMS_PER_BATCH=32`
  - `SOLVER.STAGE1.MAX_EPOCHS=60`
  - `SOLVER.STAGE2.IMS_PER_BATCH=32`
  - `SOLVER.STAGE2.MAX_EPOCHS=60`
- 产物目录：`OUTPUT/phase2/B5_i2t_w0.2_sie_olp_<machine-tag>`
- 兼容别名：`OUTPUT/today_plan_20260302_b5`（软链接到最终目录）

### P1（P0 完成后自动执行）

- 评估名：`B5_local_rebuild_rerank`
- checkpoint：`OUTPUT/phase2/B5_i2t_w0.2_sie_olp_<machine-tag>/ViT-B-16_stage2_60.pth`
- 配置：
  - baseline eval (`TEST.RE_RANKING=False`)
  - rerank eval (`--reranking`)

## 监控方式

- 主日志：`OUTPUT/phase2/B5_i2t_w0.2_sie_olp_<machine-tag>/train.log`
- eval 日志：
  - `OUTPUT/phase2/B5_i2t_w0.2_sie_olp_<machine-tag>/eval_baseline.log`
  - `OUTPUT/phase2/B5_i2t_w0.2_sie_olp_<machine-tag>/eval_rerank.log`
- 进程文件：`OUTPUT/phase2/B5_i2t_w0.2_sie_olp_<machine-tag>/launcher.pid`

## 验收标准

1. 训练进程稳定推进（日志持续输出 epoch/iteration）
2. 生成 Stage1 与 Stage2 checkpoint
3. 训练完成后自动产出 baseline + reranking 两份评估日志

---

## 实际执行结果（2026-03-02，本机）

- 机器标识：`gjob-dev-683565490869858304-taskrole1-0`
- 实验目录：`OUTPUT/phase2/B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0`
- 兼容别名：`OUTPUT/today_plan_20260302_b5`（脚本侧保留）
- 启动时间：`2026-03-02 19:52:02`
- 训练完成（Stage2 ckpt 保存）：`2026-03-02 22:46:15`
- baseline eval 完成：`2026-03-02 22:48:24`
- rerank eval 完成：`2026-03-02 22:51:31`

### 结果指标（Market-1501）

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline (`TEST.RE_RANKING=False`) | 88.3% | 94.2% | 98.2% | 98.9% |
| reranking (`--reranking`) | 91.6% | 94.3% | 97.0% | 97.6% |
| 增益（rerank - baseline） | +3.3% | +0.1% | -1.2% | -1.3% |

### 与对照基准比较（CLIP-ReID+SIE+OLP = 90.5% mAP）

- baseline：`88.3%`（较基准 `-2.2%`）
- reranking：`91.6%`（较基准 `+1.1%`）

### 验收结果

1. 训练过程稳定：已完成（日志覆盖 Stage1/Stage2 全流程）
2. checkpoint 产出：已完成（`ViT-B-16_stage1_60.pth`、`ViT-B-16_stage2_60.pth`）
3. 自动评估产出：已完成（`eval_baseline.log`、`eval_rerank.log`）

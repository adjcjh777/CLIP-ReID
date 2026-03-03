# 2026-03-03 本机新实验计划（Phase4 缺失补齐）

## 目标

基于 `docs/plans/2026-03-03-full-progress-report.md` 与 `docs/plans/2026-03-02-week-plan.md` 的未完成项，今天在本机优先补齐 Phase4 缺失消融中的 `D1_contrastive`，并在同日补齐 `D1_combined`。

本轮采用“快速筛选 + 可复现”的策略：

1. 复用已完成主线实验的 Stage1 checkpoint，跳过 Stage1，只跑 Stage2。
2. 先执行 `contrastive` 单实验（30ep），确保结果可快速回填。
3. 训练结束后自动产出 baseline 与 reranking 两份评估，便于与历史结果横向对比。

## 计划内容

### P0（立即执行）

- 实验名：`D1c_contrastive_local_s2e30`
- 数据集：`Market-1501`
- checkpoint 起点：最近一次本机 B5 Stage1
- 配置：
  - `--skip_stage1` + `--stage1_checkpoint=<B5_stage1_ckpt>`
  - `MODEL.TEXT_ENCODER_TYPE=clip_native`
  - `MODEL.TEXT_LOSS_TYPE=contrastive`
  - `MODEL.TEXT_LOSS_WEIGHT=0.5`
  - `MODEL.I2T_LOSS_WEIGHT=0.2`
  - `MODEL.SIE_CAMERA=True`
  - `MODEL.SIE_COE=3.0`
  - `MODEL.STRIDE_SIZE=[12,12]`
  - `SOLVER.STAGE2.IMS_PER_BATCH=32`
  - `SOLVER.STAGE2.MAX_EPOCHS=30`
- 产物目录：`OUTPUT/phase4/D1c_contrastive_local_s2e30_<machine-tag>`

### P1（P0 完成后自动执行）

- checkpoint：`OUTPUT/phase4/D1c_contrastive_local_s2e30_<machine-tag>/ViT-B-16_stage2_30.pth`
- baseline eval：`TEST.RE_RANKING=False`
- reranking eval：`--reranking`

## 监控方式

- 主日志：`OUTPUT/phase4/D1c_contrastive_local_s2e30_<machine-tag>/train.log`
- eval 日志：
  - `OUTPUT/phase4/D1c_contrastive_local_s2e30_<machine-tag>/eval_baseline.log`
  - `OUTPUT/phase4/D1c_contrastive_local_s2e30_<machine-tag>/eval_rerank.log`

## 验收标准

1. 训练可稳定跑满 Stage2 30 epochs。
2. 生成 `ViT-B-16_stage2_30.pth`。
3. 自动输出 baseline + reranking 两份评估日志。

---

## 实际执行结果（2026-03-03，本机）

- 状态：`已完成`
- 机器标识：`gjob-dev-683565490869858304-taskrole1-0`
- 实验目录：`OUTPUT/phase4/D1c_contrastive_local_s2e30_gjob-dev-683565490869858304-taskrole1-0`
- 启动时间：`2026-03-03 09:24:50`
- 训练完成时间（Stage2 checkpoint 保存）：`2026-03-03 10:05:14`
- baseline eval 完成：`2026-03-03 10:07:11`
- rerank eval 完成：`2026-03-03 10:10:32`

### 结果指标（Market-1501）

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline (`TEST.RE_RANKING=False`) | 86.2% | 93.1% | 97.6% | 98.3% |
| reranking (`--reranking`) | 91.0% | 93.3% | 97.1% | 97.9% |
| 增益（rerank - baseline） | +4.8% | +0.2% | -0.5% | -0.4% |

### 与已有主线结果对比（本机 B5_local_rebuild）

- 对比本机 B5 baseline（88.3%）：本实验 baseline 下降 `-2.1%`。
- 对比本机 B5 reranking（91.6%）：本实验 reranking 下降 `-0.6%`。
- 对比统一对照基准（90.5%）：本实验 reranking 仍高于基准 `+0.5%`。

### 结论

1. `contrastive` 在本轮配置下对 baseline 不利（86.2%），不优于 `none`。
2. reranking 仍能提供显著增益（+4.8% mAP），但最终绝对值仍低于本机 B5+reranking。
3. Phase4 缺失项中的 `D1_combined` 已在同日执行完成，结果见下方补充。

---

## 实际执行结果（2026-03-03，本机，D1_combined）

- 状态：`已完成`
- 机器标识：`gjob-dev-683565490869858304-taskrole1-0`
- 实验目录：`OUTPUT/phase4/D1c_combined_local_s2e30_gjob-dev-683565490869858304-taskrole1-0`
- 启动时间：`2026-03-03 10:17:17`
- 训练完成时间（Stage2 checkpoint 保存）：`2026-03-03 10:57:34`
- baseline eval 完成：`2026-03-03 10:59:33`
- rerank eval 完成：`2026-03-03 11:04:57`

### 结果指标（Market-1501）

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline (`TEST.RE_RANKING=False`) | 86.4% | 93.3% | 97.7% | 98.5% |
| reranking (`--reranking`) | 90.8% | 93.4% | 96.6% | 97.6% |
| 增益（rerank - baseline） | +4.4% | +0.1% | -1.1% | -0.9% |

### 对比结论（Phase4）

1. `combined` baseline（86.4%）略高于 `contrastive` baseline（86.2%），但两者都低于本机 B5 baseline（88.3%）。
2. reranking 后 `combined` 为 90.8%，低于 `contrastive` reranking（91.0%）和本机 B5+reranking（91.6%）。
3. Phase4 缺失项已补齐，当前可得结论是 `TEXT_LOSS_TYPE=none` 仍为更优选。

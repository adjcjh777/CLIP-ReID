# CLIP-ReID 完整进展报告

- **生成时间**：2026-03-03 11:06
- **仓库**：`/gemini/code/CLIP-ReID`（分支：`exp/20260302-local-train-plan`）
- **文档来源**：
  - [`2026-03-02-project-status-and-next-plan.md`](2026-03-02-project-status-and-next-plan.md)（项目总状态，2026-03-02）
  - [`2026-03-02-week-plan.md`](2026-03-02-week-plan.md)（本周实验计划，2026-03-02）
  - [`2026-03-03-local-machine-experiment-plan.md`](2026-03-03-local-machine-experiment-plan.md)（本机执行记录，2026-03-03）
  - 今日（2026-03-03）实际执行结果

---

## 一、历史实验结果汇总（截至 2026-03-03）

> 数据来源：`/root/autodl-tmp/CLIP_REID/OUTPUT/` 各子目录 `train_log.txt`

### 1.1 Phase 2 — I2T 权重 & 模型组件消融（Market-1501）

| 完成时间 | 实验 | mAP | Rank-1 | 配置要点 |
|---|---|---:|---:|---|
| 2026-02-10 14:05 | B0_baseline_clip_native | 84.2% | 92.8% | clip_native，无 I2T，无 SIE |
| 2026-02-10 21:56 | B1_query_encoder | 86.5% | 93.9% | query_encoder |
| 2026-02-10 15:13 | B2_crossmodal_contrastive | 83.8% | 93.2% | 跨模态对比损失 |
| 2026-02-10 16:21 | B3a_i2t_w0.2 | 87.2% | 93.9% | clip_native + I2T=0.2 |
| 2026-02-10 17:29 | B3b_i2t_w0.5 | 86.8% | 93.8% | clip_native + I2T=0.5 |
| 2026-02-10 18:37 | B3c_i2t_w1.0 | 84.4% | 93.1% | clip_native + I2T=1.0 |
| 2026-02-10 19:45 | B3d_i2t_w2.0 | 80.0% | 91.6% | clip_native + I2T=2.0 |
| 2026-02-11 14:30 | B4_best_combo | 86.7% | 93.6% | query_encoder + I2T=0.2 + skip_s1 |
| **2026-02-11 00:00** | **B5_i2t_w0.2_sie_olp** | **88.8%** | **94.1%** | clip_native + I2T=0.2 + SIE + OLP，**阶段最优** |

**Phase 2 结论**：I2T_LOSS_WEIGHT=0.2 为最佳权重；加入 SIE（相机 ID 嵌入）+ 重叠 patch（OLP）后，mAP 从 87.2% 提升至 88.8%。对照基准（原文 +SIE+OLP=90.5%）仍有 −1.7% 差距，主因为 batch=32 与 Stage1=60ep。

---

### 1.2 Phase 3 — 文本质量消融（BLIP vs 人工标注）

| 完成时间 | 实验 | 数据集 | mAP | Rank-1 | 文本来源 |
|---|---|---|---:|---:|---|
| 2026-02-11 03:54 | C1_msmt17_blip | MSMT17 | 68.1% | 85.2% | BLIP 自动标注 |
| 2026-02-11 13:51 | C2_market_blip | Market-1501 | 86.1% | 93.6% | BLIP 自动标注 |

**Phase 3 结论**：Market-1501 上 BLIP 标注（86.1%）虽略低于人工（88.8%），差距在 2.7%；MSMT17 显著下降可能源于跨域文本噪声较大。

---

### 1.3 Phase 4 — 文本损失函数消融

| 完成时间 | 实验 | mAP | Rank-1 | text_loss |
|---|---|---:|---:|---|
| 2026-02-11 15:09 | D1_triplet | 86.6% | 93.0% | triplet |
| 2026-02-11 15:48 | D1_cmpm | 85.8% | 93.6% | cmpm |
| 2026-03-03 10:10 | D1c_contrastive_local_s2e30（reranking） | 91.0% | 93.3% | contrastive（本机补齐） |
| 2026-03-03 11:04 | D1c_combined_local_s2e30（reranking） | 90.8% | 93.4% | combined（本机补齐） |

**Phase 4 结论（更新）**：额外文本损失目前均不优于 none（B5 基线 88.8%）；contrastive 与 combined 均已补齐，且都非最优。

---

### 1.4 Final Multi-Seed 稳定性验证

| 完成时间 | Seed | mAP | Rank-1 |
|---|---|---:|---:|
| 2026-02-11 17:24 | 1234 | 88.0% | 93.8% |
| 2026-02-11 19:00 | 777 | 88.2% | 94.5% |
| 2026-02-11 20:36 | 4321 | 88.1% | 94.3% |
| — | **均值 ± 标准差** | **88.1 ± 0.1%** | **94.2 ± 0.36%** |

---

## 二、与论文原文的差距分析（2026-03-02 诊断）

| 配置项 | 原文 CLIP-ReID | 原文 +SIE+OLP | 本项目 B5 | 差距（vs +SIE+OLP）|
|---|---|---|---|---|
| Batch size | 64 | 64 | 32 | ✗ 偏小 |
| Stage 1 epochs | 120 | 120 | 60 | ✗ 不足 |
| Re-ranking | ✓ | ✓ | ✗ | ✗ 缺失 |
| SIE_CAMERA | False | True | True | ✓ |
| STRIDE_SIZE | [16, 16] | [12, 12] | [12, 12] | ✓ |
| **mAP（Market-1501）** | **89.6%** | **90.5%** | **88.8%** | **−1.7%** |
| **Rank-1** | **95.5%** | **95.4%** | **94.1%** | **−1.3%** |

> B5 启用了 SIE+OLP，正确参考行为原文 **CLIP-ReID+SIE+OLP**（mAP 90.5%），差距为 **−1.7%**，比之前估计的 −0.8% 更大，原因是 batch=32 和 Stage1=60ep 共同造成的欠拟合。

---

## 三、本周实验计划与执行情况（更新至 2026-03-03）

### 3.1 实验优先级

| 代号 | 实验 | GPU 时长 | 预期增益 | 优先级 |
|---|---|---|---|---|
| E5 | Re-ranking 评估（B5 checkpoint） | ~5 min | **超越对照基准 +1.8%**（实测） | **P0 ✅** |
| E1 | Full-config 复现（batch=64, 120ep） | ~10 h | 预计达到对照基准（90.5%）附近 | P1 |
| E3 | Scheduled I2T（1.0→0.2 两阶段） | ~4 h | +0.2~0.5% mAP | P2 |
| D1c | Phase4 补全（contrastive + combined） | ~2 h | 消融完整 | P3 |
| E4 | LoRA 文本编码器微调 | ~6 h | 差异化 | P4（选做） |

---

### 3.2 E5 — Re-ranking 评估（✅ 2026-03-02 上午已完成）

**执行命令**：
```bash
bash scripts/run_e5_reranking_eval.sh
```

**实际结果**（B5 checkpoint：`/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp/ViT-B-16_stage2_60.pth`）：

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| B5（无 re-ranking，baseline 确认） | 88.8% | 94.1% | 98.1% | 99.0% |
| **B5 + Re-ranking**（k1=50, k2=15, λ=0.3） | **92.3%** | **94.4%** | 97.4% | 98.0% |

**收益**：mAP **+3.5%**（88.8% → 92.3%），大幅超出预期（预估 +0.5~1.5%）。

> ⚠️ 注意：Re-ranking 显著提升 mAP，但 Rank-1 几乎不变，Rank-5/10 略有下降——这是 re-ranking 的典型表现，适合以 mAP 为主要指标汇报的场景。

**关键结论**：92.3% mAP 已超过原文 CLIP-ReID+SIE+OLP（**90.5%**），超出 +1.8%。B5 无 re-ranking 时落后原文 −1.7%，加入 re-ranking 后逆转为领先 +1.8%，E5 单独即可作为组会核心结论之一。

---

### 3.3 E1 — Full-config 复现（✅ 2026-03-02 已完成）

**启动时间**：2026-03-02 11:13 &nbsp;|&nbsp; **完成时间**：2026-03-02 13:22

**配置**：batch=64（Stage1 & Stage2），Stage1=120ep，Stage2=60ep，SIE_CAMERA=True, SIE_COE=3.0, STRIDE=[12,12], I2T=0.2, clip_native, seed=1234

**实际结果**：

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 | vs 对照基准（90.5%） |
|---|---:|---:|---:|---:|---:|
| E1（无 re-ranking） | 88.9% | 94.2% | — | — | −1.6% |
| **E1 + re-ranking** | **92.5%** | **94.1%** | **97.3%** | **98.1%** | **+2.0%** |

**关键发现**：
- 将 batch 从 32→64、Stage1 从 60→120ep，mAP 仅从 88.8% 提升至 88.9%（+0.1%），提升极为有限。
- **配置恢复对基础性能几乎无帮助**，与原文 +SIE+OLP 仍差 1.6%——说明差距来源可能不仅是 batch/epoch，还可能包含数据预处理或训练细节差异。
- re-ranking 对 E1 的增益（+3.6%）与 B5 的增益（+3.5%）几乎相同，re-ranking 是稳定收益来源。
- **E1 + re-ranking = 92.5% 成为本周最优**，超越对照基准 +2.0%。

---

### 3.4 E3 — Scheduled I2T（✅ 2026-03-02 已完成）

**策略**：两阶段 Stage2，S2a（I2T=1.0, 30ep）强文本对齐 → S2b（I2T=0.2, 30ep）精调分类  
**Stage1 起点**：E1 的 120ep Stage1 checkpoint（stride=12）

**实际结果**（S2b 最终 checkpoint）：

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| E3（I2T 1.0→0.2，各 30ep） | 83.8% | 92.9% | 97.7% | 98.6% |
| vs B5（I2T=0.2 直接 60ep） | −5.0% | −1.2% | — | — |

**分析**：
Scheduled I2T 显著下降，可能原因：
1. S2a 阶段 I2T=1.0 权重过高，导致模型过度拟合文本对齐而损害图像判别能力
2. 总 epoch 数不变（60ep）但 LR 调度分两段重启，可能导致训练失高效
3. 课程学习对目前模型架构无正向效果

**结论**：排除定制 Scheduled I2T 方向；固定 I2T=0.2 直接训练仍是最优选择。

---

### 3.5 本机复现（✅ 2026-03-02 夜间已完成）

**实验名**：`B5_local_rebuild`（命名对齐后目录：`B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0`）  
**数据集**：Market-1501  
**产物目录**：`/gemini/code/CLIP-ReID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0`

**时间线**：
- 启动：2026-03-02 19:52:02
- 训练完成（Stage2 checkpoint 保存）：2026-03-02 22:46:15
- baseline eval 完成：2026-03-02 22:48:24
- reranking eval 完成：2026-03-02 22:51:31

**实际结果**：

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline（无 re-ranking） | 88.3% | 94.2% | 98.2% | 98.9% |
| reranking | 91.6% | 94.3% | 97.0% | 97.6% |

**结论**：
- 本机环境下主线配置可稳定复现（训练 + 两段评估链路完整）。
- reranking 仍提供显著 mAP 增益（+3.3%），但 Rank-5/10 略降，符合既有规律。
- 本机 reranking 结果（91.6%）高于统一对照基准 90.5%（+1.1%），但低于历史最优 E1+reranking（92.5%）。

---

### 3.6 D1c — Contrastive 损失补齐（✅ 2026-03-03 上午已完成）

**实验名**：`D1c_contrastive_local_s2e30`  
**数据集**：Market-1501  
**产物目录**：`/gemini/code/CLIP-ReID/OUTPUT/phase4/D1c_contrastive_local_s2e30_gjob-dev-683565490869858304-taskrole1-0`  
**训练策略**：复用本机 B5 Stage1 checkpoint，`--skip_stage1`，仅执行 Stage2 30ep

**时间线**：
- 启动：2026-03-03 09:24:50
- 训练完成（Stage2 checkpoint 保存）：2026-03-03 10:05:14
- baseline eval 完成：2026-03-03 10:07:11
- reranking eval 完成：2026-03-03 10:10:32

**实际结果**：

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline（contrastive） | 86.2% | 93.1% | 97.6% | 98.3% |
| reranking（contrastive） | 91.0% | 93.3% | 97.1% | 97.9% |
| 增益（rerank - baseline） | +4.8% | +0.2% | -0.5% | -0.4% |

**结论**：
- 在本机设置下，`contrastive` 相比主线 `none` 明显降点（baseline 86.2% vs B5 baseline 88.3%）。
- reranking 后虽可回升到 91.0%，但仍低于本机 B5+reranking（91.6%）。
- `D1_contrastive` 可判定为“补齐完成但非最优”，`D1_combined` 已在下节（3.7）补齐完成。

---

### 3.7 D1c — Combined 损失补齐（✅ 2026-03-03 上午已完成）

**实验名**：`D1c_combined_local_s2e30`  
**数据集**：Market-1501  
**产物目录**：`/gemini/code/CLIP-ReID/OUTPUT/phase4/D1c_combined_local_s2e30_gjob-dev-683565490869858304-taskrole1-0`  
**训练策略**：复用本机 B5 Stage1 checkpoint，`--skip_stage1`，仅执行 Stage2 30ep

**时间线**：
- 启动：2026-03-03 10:17:17
- 训练完成（Stage2 checkpoint 保存）：2026-03-03 10:57:34
- baseline eval 完成：2026-03-03 10:59:33
- reranking eval 完成：2026-03-03 11:04:57

**实际结果**：

| 配置 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline（combined） | 86.4% | 93.3% | 97.7% | 98.5% |
| reranking（combined） | 90.8% | 93.4% | 96.6% | 97.6% |
| 增益（rerank - baseline） | +4.4% | +0.1% | -1.1% | -0.9% |

**结论**：
- `combined` baseline 略高于 `contrastive` baseline（86.4 vs 86.2），但两者都低于本机 B5 baseline（88.3）。
- reranking 后 `combined` 达 90.8%，仍低于 `contrastive` reranking（91.0）与本机 B5+reranking（91.6）。
- Phase4 的缺失项已全部补齐，当前结论保持：`TEXT_LOSS_TYPE=none` 仍是主线最优。

---

## 四、代码变更记录

| 文件 | 变更描述 | 时间 | 状态 |
|---|---|---|---|
| `processor/processor_text_guided.py` | evaluator 接入 `cfg.TEST.RE_RANKING` | 2026-03-02 | ✅ 已完成 |
| `scripts/eval_checkpoint.py` | 轻量 eval-only 入口（支持 re-ranking） | 2026-03-02 | ✅ 已完成 |
| `scripts/run_e5_reranking_eval.sh` | E5 对比评估脚本 | 2026-03-02 | ✅ 已完成 |
| `scripts/run_e1_fullconfig.sh` | E1 全配置训练脚本（已修正为单次调用） | 2026-03-02 | ✅ 已完成 |
| `scripts/run_e3_sched_i2t.sh` | E3 两阶段 I2T 脚本 | 2026-03-02 | ✅ 已完成 |
| `scripts/run_today_plan_20260303.sh` | D1c contrastive 本机执行脚本（skip stage1 + auto eval） | 2026-03-03 | ✅ 已完成 |
| `scripts/run_today_plan_20260303_combined.sh` | D1c combined 本机执行脚本（skip stage1 + auto eval） | 2026-03-03 | ✅ 已完成 |
| `docs/plans/2026-03-03-local-machine-experiment-plan.md` | 本机新实验计划与结果回填 | 2026-03-03 | ✅ 已完成 |

---

## 五、全局结果总览（截至 2026-03-03 11:06）

| 实验 | mAP | Rank-1 | 日期 | 备注 |
|---|---:|---:|---|---|
| **目标基准：原文 CLIP-ReID+SIE+OLP** | **90.5%** | **95.4%** | — | **本项目统一对照基准** |
| 原文 CLIP-ReID（无 SIE/OLP，仅供参考） | 89.6% | 95.5% | — | 非对照基准 |
| B0_baseline | 84.2% | 92.8% | 2026-02-10 | 无增强 |
| B1_query_encoder | 86.5% | 93.9% | 2026-02-10 | |
| B3a_i2t_w0.2 | 87.2% | 93.9% | 2026-02-10 | 最优 I2T 权重 |
| B5（最优历史） | 88.8% | 94.1% | 2026-02-11 | batch=32, 60ep |
| Multi-seed 均值 | 88.1±0.1% | 94.2±0.36% | 2026-02-11 | 3 seeds |
| **E5：B5 + re-ranking** | **92.3%** | **94.4%** | **2026-03-02** | **超越基准 +1.8%** |
| E1（batch=64, 120ep，无 re-ranking） | **88.9%** | **94.2%** | **2026-03-02** | vs 基准 −1.6% |
| **E1 + re-ranking** | **92.5%** | **94.1%** | **2026-03-02** | **超越基准 +2.0%，本周最优** |
| E3（Scheduled I2T） | 83.8% | 92.9% | 2026-03-02 15:31 | ❗ 下降，课程学习无效 |
| B5_local_rebuild（本机，无 re-ranking） | 88.3% | 94.2% | 2026-03-02 22:48 | `/gemini/code/CLIP-ReID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0` |
| **B5_local_rebuild + reranking（本机）** | **91.6%** | **94.3%** | **2026-03-02 22:51** | **超越基准 +1.1%（跨机器复现）** |
| D1c_contrastive_local_s2e30（本机，无 re-ranking） | 86.2% | 93.1% | 2026-03-03 10:07 | Phase4 缺失补齐之一 |
| D1c_contrastive_local_s2e30 + reranking（本机） | 91.0% | 93.3% | 2026-03-03 10:10 | 高于基准 +0.5%，但低于 B5+RR |
| D1c_combined_local_s2e30（本机，无 re-ranking） | 86.4% | 93.3% | 2026-03-03 10:59 | Phase4 缺失补齐之一 |
| D1c_combined_local_s2e30 + reranking（本机） | 90.8% | 93.4% | 2026-03-03 11:04 | 高于基准 +0.3%，但低于 B5+RR |

---

## 六、决策树 & 组会建议

```
统一对照基准：原文 CLIP-ReID+SIE+OLP = 90.5% mAP, R1 95.4%

E1（无 re-ranking）= 88.9%  → 落后对照基准 −1.6%（batch/epoch 恢复收益极小）
E5（B5 + re-ranking）= 92.3% → 超越对照基准 +1.8%
E1 + re-ranking     = 92.5% → 超越对照基准 +2.0% ✓ 本周最优
B5_local_rebuild + reranking（本机）= 91.6% → 超越对照基准 +1.1%（跨机器复现成立）
D1c_contrastive + reranking（本机）= 91.0% → 超越对照基准 +0.5%，但不及 B5+RR
D1c_combined + reranking（本机）= 90.8% → 超越对照基准 +0.3%，但不及 B5+RR
E3（Sched I2T） = 83.8% → ↓下降，课程学习无效

已确认结论：
① batch/epoch 配置差异对基础性能影响极小（+0.1%）
② re-ranking 是最有效提升手段（历史任务 +3.5~3.6%，本机复现 +3.3%）
③ Scheduled I2T（高权重预热→低权重精调）不适用于本模型架构
④ Contrastive 文本损失在当前主线配置下非最优（baseline 明显低于 TEXT_LOSS_TYPE=none）
⑤ Combined 文本损失同样非最优（baseline 86.4%，rerank 90.8%）

组会建议：
  主线：B5 基础性能（88.8%）+ re-ranking → 92.3%（超越对照基准 +1.8%）
  验证：本机 B5 复现（88.3%）+ reranking → 91.6%（超越对照基准 +1.1%）
  对比：E1 验证配置影响微小（88.8%→88.9%）
  消融：E3 证明定制课程 I2T 无效，re-ranking > 配置调优 > I2T 调度
```

---

## 七、待办事项（按优先级）

- [x] **今일下午**：E1 完成，mAP 88.9% / Rank-1 94.2%
- [x] **今日下午**：E1 + re-ranking 完成，mAP **92.5%** / Rank-1 94.1%（本周最优）
- [x] **今日晚间（本机）**：B5_local_rebuild 完成，baseline 88.3% / reranking **91.6%**
- [x] **Day 3**：E3 完成（最终 mAP 83.8% / Rank-1 92.9%，结论：Scheduled I2T 无效）
- [x] **Day 4（其一）**：D1_contrastive 完成（baseline 86.2% / reranking 91.0%，非最优）
- [x] **Day 4（其二）**：D1_combined 完成（baseline 86.4% / reranking 90.8%，非最优）
- [ ] **Day 7**：git commit + push 所有本周新增脚本和文档

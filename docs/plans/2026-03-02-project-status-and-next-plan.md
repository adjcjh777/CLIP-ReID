# CLIP-ReID 项目现状与后续执行计划（按时间戳整理）

- 生成时间：2026-03-03 11:15（含 D1c_combined 完成后更新）
- 仓库：`/gemini/code/CLIP-ReID`
- 证据来源：
  - Git 分支与提交时间戳
  - `/root/autodl-tmp/CLIP_REID` 下训练日志、pipeline 日志、注释文件、汇总表
  - `/gemini/code/CLIP-ReID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0` 本机训练与评估日志

---

## 1) 当前总体结论（TL;DR）

1. **历史主线分支是 `exp/text-guided-reid`**；当前本机执行分支为 `exp/20260302-local-train-plan`，训练链路已在新机器跑通。
2. `autodl-tmp` 证据显示：**Phase2/3/4 与 final multiseed 核心实验在 2026-02-11 当天串行跑完**（pipeline done 标记完整）。
3. **本机复现主线已完成**：`B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0` 在 2026-03-02 夜间完成，baseline mAP 88.3%，reranking mAP 91.6%。
4. 当前主要不是“缺训练结果”，而是**缺收口动作**：
   - 结果回填与结论固化
   - 本地改动整理提交/推送
   - Phase4 缺失项已在 2026-03-03 本机全部补齐（`contrastive` + `combined` 均完成）

---

## 2) `autodl-tmp` 文档/产物读取结论（关键项）

> 说明：`autodl-tmp` 下大多数是训练日志；可作为“文档证据”的关键文件如下。

### 2.1 关键证据文件（按时间）

- 2026-01-25 12:35:32
  - `/root/autodl-tmp/CLIP_REID/OUTPUT/summary/summary_20260125_115045_final.xlsx`
  - 含阶段性汇总（早期网格/seed实验）
- 2026-02-11 14:30:09 ~ 20:36:38
  - `/root/autodl-tmp/CLIP_REID/OUTPUT/pipeline/*.log`
  - `/root/autodl-tmp/CLIP_REID/OUTPUT/pipeline/.p*.done`
  - 证明 B4 → D1_triplet → D1_cmpm → seed_1234/777/4321 顺序执行完成
- 2026-02-11 00:44:45 / 00:54:59
  - `/root/CLIP-ReID/annotations/msmt17_train_blip.json`
  - `/root/CLIP-ReID/annotations/market1501_train_blip.json`
  - BLIP 注释已实际生成（后续 C1/C2 使用）

### 2.2 Phase2/3/4 + Final 的关键结果（来自 train_log 时间戳）

| 时间 | 实验 | mAP | Rank-1 | 证据 |
|---|---|---:|---:|---|
| 2026-02-10 14:05 | B0_baseline_clip_native | 84.2 | 92.8 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B0_baseline_clip_native/train_log.txt` |
| 2026-02-10 21:56 | B1_query_encoder | 86.5 | 93.9 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B1_query_encoder/train_log.txt` |
| 2026-02-10 15:13 | B2_crossmodal_contrastive | 83.8 | 93.2 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B2_crossmodal_contrastive/train_log.txt` |
| 2026-02-10 16:21 | B3a_i2t_w0.2 | 87.2 | 93.9 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B3a_i2t_w0.2/train_log.txt` |
| 2026-02-10 17:29 | B3b_i2t_w0.5 | 86.8 | 93.8 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B3b_i2t_w0.5/train_log.txt` |
| 2026-02-10 18:37 | B3c_i2t_w1.0 | 84.4 | 93.1 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B3c_i2t_w1.0/train_log.txt` |
| 2026-02-10 19:45 | B3d_i2t_w2.0 | 80.0 | 91.6 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B3d_i2t_w2.0/train_log.txt` |
| 2026-02-11 00:00 | B5_i2t_w0.2_sie_olp | **88.8** | **94.1** | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp/train_log.txt` |
| 2026-02-11 03:54 | C1_msmt17_blip | 68.1 | 85.2 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase3/C1_msmt17_blip/train_log.txt` |
| 2026-02-11 13:51 | C2_market_blip | 86.1 | 93.6 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase3/C2_market_blip/train_log.txt` |
| 2026-02-11 14:30 | B4_best_combo | 86.7 | 93.6 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B4_best_combo/train_log.txt` |
| 2026-02-11 15:09 | D1_triplet | 86.6 | 93.0 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase4/D1_triplet/train_log.txt` |
| 2026-02-11 15:48 | D1_cmpm | 85.8 | 93.6 | `/root/autodl-tmp/CLIP_REID/OUTPUT/phase4/D1_cmpm/train_log.txt` |
| 2026-02-11 17:24 | final seed_1234 | 88.0 | 93.8 | `/root/autodl-tmp/CLIP_REID/OUTPUT/final_multiseed/seed_1234/train_log.txt` |
| 2026-02-11 19:00 | final seed_777 | 88.2 | 94.5 | `/root/autodl-tmp/CLIP_REID/OUTPUT/final_multiseed/seed_777/train_log.txt` |
| 2026-02-11 20:36 | final seed_4321 | 88.1 | 94.3 | `/root/autodl-tmp/CLIP_REID/OUTPUT/final_multiseed/seed_4321/train_log.txt` |

- final multiseed 统计：
  - mAP: `88.1 ± 0.1`
  - Rank-1: `94.2 ± 0.36`

### 2.3 本机补充证据（2026-03-02 夜间）

| 时间 | 实验 | mAP | Rank-1 | 证据 |
|---|---|---:|---:|---|
| 2026-03-02 22:48 | B5_local_rebuild（baseline） | 88.3 | 94.2 | `/gemini/code/CLIP-ReID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0/eval_baseline.log` |
| 2026-03-02 22:51 | B5_local_rebuild + reranking | **91.6** | **94.3** | `/gemini/code/CLIP-ReID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp_gjob-dev-683565490869858304-taskrole1-0/eval_rerank.log` |

- 本机复现结论：
  - 训练链路与评估链路均可稳定运行（checkpoint 与两份 eval 日志完整）。
  - reranking 相对 baseline 带来 mAP `+3.3`，与历史趋势一致。
  - reranking 结果高于统一对照基准 90.5（`+1.1`）。

### 2.4 本机补充证据（2026-03-03 上午，D1c）

| 时间 | 实验 | mAP | Rank-1 | 证据 |
|---|---|---:|---:|---|
| 2026-03-03 10:07 | D1c_contrastive_local_s2e30（baseline） | 86.2 | 93.1 | `/gemini/code/CLIP-ReID/OUTPUT/phase4/D1c_contrastive_local_s2e30_gjob-dev-683565490869858304-taskrole1-0/eval_baseline.log` |
| 2026-03-03 10:10 | D1c_contrastive_local_s2e30 + reranking | **91.0** | **93.3** | `/gemini/code/CLIP-ReID/OUTPUT/phase4/D1c_contrastive_local_s2e30_gjob-dev-683565490869858304-taskrole1-0/eval_rerank.log` |
| 2026-03-03 10:59 | D1c_combined_local_s2e30（baseline） | 86.4 | 93.3 | `/gemini/code/CLIP-ReID/OUTPUT/phase4/D1c_combined_local_s2e30_gjob-dev-683565490869858304-taskrole1-0/eval_baseline.log` |
| 2026-03-03 11:04 | D1c_combined_local_s2e30 + reranking | **90.8** | **93.4** | `/gemini/code/CLIP-ReID/OUTPUT/phase4/D1c_combined_local_s2e30_gjob-dev-683565490869858304-taskrole1-0/eval_rerank.log` |

- D1c（contrastive + combined）结论：
  - baseline：`combined`（86.4）略高于 `contrastive`（86.2），但两者都低于本机 B5 baseline（88.3）。
  - reranking：`contrastive`=91.0、`combined`=90.8，均高于对照基准 90.5，但都低于本机 B5+reranking（91.6）。
  - Phase4 补齐状态更新为：`contrastive` 与 `combined` 均已完成，当前主线最优结论仍为 `TEXT_LOSS_TYPE=none`。

---

## 3) 分支进展情况（Git）

## 3.1 分支规模与活跃度（相对 master）

| 分支 | 最新提交时间 | ahead(master) | behind(master) | 变更文件数 |
|---|---|---:|---:|---:|
| master | 2023-11-21 | 0 | 0 | - |
| exp/market_clip | 2026-01-25 | 19 | 0 | 27 |
| feature/multi-granularity-fusion | 2026-01-27 | 22 | 0 | 38 |
| exp/text-guided-reid | 2026-02-10 | 17 | 0 | 38 |

结论：三个实验分支都不落后 master，且互相并非祖先链关系，属于并行分叉研发。

## 3.2 与远端同步状态（本地缓存的 origin 引用）

| 分支 | 相对 origin ahead | 相对 origin behind |
|---|---:|---:|
| exp/market_clip | 4 | 0 |
| exp/text-guided-reid | 12 | 0 |
| feature/multi-granularity-fusion | 3 | 0 |
| master | 0 | 0 |

> 注意：当前环境 `git fetch` 受代理影响失败（127.0.0.1:10808），上表基于本地已有远端引用缓存。

## 3.3 当前工作区状态（正在 `exp/text-guided-reid`）

- 已跟踪但未提交：
  - `datasets/text_reid_dataset.py`
  - `docs/experiments/phase2_ablation_results.xlsx`
  - `docs/plans/plan-textGuidedReidExperiments.prompt.md`
  - `scripts/generate_captions_blip.py`
  - `scripts/run_phase2_ablation.sh`
  - `scripts/update_experiment_results.py`
- 未跟踪新增：
  - `annotations/market1501_train_blip.json`
  - `annotations/msmt17_train_blip.json`
  - `scripts/run_b4_best_combo.sh`
  - `scripts/run_multiseed_final.sh`
  - `scripts/run_phase3.sh`
  - `scripts/run_phase4_loss_ablation.sh`
  - `scripts/run_priority_pipeline.sh`
  - `scripts/watch_and_shutdown.sh`

结论：代码与实验资产已经累计完成，但还处于“本地沉淀未收口”的状态。

---

## 4) 时间线梳理（关键里程碑）

- **2026-01-12 ~ 2026-01-26**：
  - 大量基线、seed、pct、SIE/OLP、多粒度实验日志落地（Market1501 / MSMT17）。
  - 形成早期 summary 汇总表（`summary_20260125_115045_final.xlsx`）。
- **2026-01-27**：
  - text-guided 与 multi-granularity 两条研发线并行推进（从 Git 提交与 `OUTPUT/text_guided/*` 可证）。
- **2026-02-07 ~ 2026-02-10**：
  - text-guided 代码打通（query encoder、loss、评估脚本、实验脚本）并开始 Phase2 系统实验。
- **2026-02-11（高密度收敛日）**：
  1) 凌晨完成 B5（当前全局最优）与 C1
  2) 中午完成 C2
  3) 下午完成 B4、D1_triplet、D1_cmpm
  4) 晚间完成 final multiseed 三个 seed
  5) pipeline done 标记全部就位

---

## 5) 现在的“未完成事项”定义（按目标口径）

### 5.1 如果目标是“工程收口”

未完成项：
1. 将本地未提交改动整理为可审阅提交（按功能拆分 commit）。
2. 推送三个分支未发布提交（至少先推 `exp/text-guided-reid`）。
3. 把 C2/D1/final multiseed + 本机复现实验（已产出）统一回填到计划文档与汇总表，形成最终结论版文档（Phase4 缺失项已补齐）。

### 5.2 如果目标是“实验完整性（论文级消融）”

未完成项：
1. 统一形成 D1 全量对比表（none/contrastive/triplet/cmpm/combined），并沉淀为最终版结论图表。
2. 对本机与 autodl 两套环境结果给出并排结论说明（避免后续复盘歧义）。
3. 若追求更强说服力：补充 C1/C2 的误检案例可视化与文本质量分析。

---

## 6) 接下来执行计划（建议按 1 周）

## Day 1（收口优先）

1. 清理并提交 `exp/text-guided-reid` 本地改动：
   - Commit A：训练/实验脚本（`scripts/*phase*`, `run_priority_pipeline.sh`, `watch_and_shutdown.sh`）
   - Commit B：数据与标注生成逻辑（`generate_captions_blip.py`, `datasets/text_reid_dataset.py`, `annotations/*_blip.json`）
   - Commit C：文档与台账（`plan-textGuidedReidExperiments.prompt.md`, `phase2_ablation_results.xlsx`）
2. 推送到远端并打一个里程碑 tag（如 `text-guided-phase2-4-20260211`）。

## Day 2-3（归档 Phase4 结论）

1. D1_contrastive 与 D1_combined 两组实验已完成（本机 2026-03-03 上午），本阶段仅做结果归档与结论固化。
2. 统一输出 D1 全量表：
   - 指标：mAP、R1、训练时长、稳定性（若可）
3. 更新总结：确认“最佳损失配置”是否仍为 `none + I2T=0.2 + SIE/OLP` 或出现新最优。

## Day 4（文档与可复现）

1. 在 `docs/experiments` 写最终实验纪要：
   - 实验设置、时间线、结论、失败案例
2. 固化一键执行入口：
   - 保留 `run_priority_pipeline.sh`，补 readme 用法与断点续跑说明

## Day 5（分支治理）

1. `exp/market_clip` 与 `feature/multi-granularity-fusion` 分别整理未推送提交并推送。
2. 决定后续主干策略：
   - 方案 A：继续各分支并行
   - 方案 B：将稳定工具链择优 cherry-pick 到 text-guided 主线

---

## 7) 本周执行优先级（建议）

- **P0（必须）**：提交/推送 `exp/text-guided-reid` + 更新最终文档台账
- **P1（高）**：归档 Phase4 全量损失消融结论
- **P2（中）**：可视化分析与报告整理
- **P3（中）**：另外两个分支的远端同步与治理

---

## 8) 一句话状态判断

项目当前不缺核心结果，且已完成跨机器复现与 Phase4 补齐；**缺的是把“2 月 11 日历史闭环 + 3 月 2-3 日本机复现”统一固化为可复现、可汇报、可合并的工程资产**。

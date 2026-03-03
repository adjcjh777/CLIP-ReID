## Plan: Text-Guided ReID 机制完善实验计划

**TL;DR** — 当前系统已跑通 text-guided 训练管线并在 Market1501 上达到 88.0% mAP（vs 基线 89.6%），但存在三大未闭合模块：`TextQueryEncoder` 未接入训练、跨模态损失（Triplet+CMPM）仅部分使用、MSMT17 标注只是占位文本。本计划在 2-3 周内分 4 个阶段系统打通这些模块，并通过消融实验验证每个组件的贡献。考虑到 AutoDL 按需计费，实验以"单变量控制 + 高收益优先"原则排列，预估总 GPU 时间 ~80-100 小时（以 3090 为例）。

---

**Steps**

### 阶段 1：代码修复与管线打通（Day 1-3，纯代码，无需 GPU）

1. **修复已知代码缺陷**
   - 修正 loss/make_loss.py 中硬编码的 `feat_dim=2048` → 从 cfg 读取（ViT 应为 768）
   - 修正 loss/triplet_loss.py 中错误导入 `from turtle import pd`
   - 修正 scripts/evaluate_text_reid.py 中 `build_text_reid_dataloaders` 的接口不匹配问题

2. **接入 `TextQueryEncoder` 到训练流程**
   - 在 model/make_model_clipreid.py 中将 `TextQueryEncoder`（已定义于 model/text_query_encoder.py）作为可选文本编码路径，通过 config 控制 `MODEL.TEXT_ENCODER_TYPE` 切换 `encode_text_tokens`（CLIP 原生）与 `TextQueryEncoder`（属性注意力增强）
   - 在 processor/processor_text_guided.py 的 Stage 1 和 Stage 2 中增加 `TextQueryEncoder` 的前向调用分支
   - 在 config/defaults.py 新增 `MODEL.TEXT_ENCODER_TYPE = 'clip_native'`（可选值：`clip_native` / `query_encoder`）和 `MODEL.TEXT_TEMPERATURE = 0.07`

3. **接入完整跨模态损失到 Stage 2**
   - 将 loss/cross_modal_loss.py 中的 `CrossModalContrastiveLoss` 也接入 Stage 2（当前仅 Stage 1 使用）
   - 通过 `make_text_reid_loss(cfg)` 工厂函数，将 `TextImageTripletLoss` 和 `CMPMLoss` 注册为可组合选项
   - 在 config/defaults.py 新增 `MODEL.TEXT_LOSS_TYPE = 'contrastive'`（可选：`contrastive` / `triplet` / `cmpm` / `combined`）和 `MODEL.TEXT_LOSS_WEIGHT = 1.0`

4. **打通 Text-to-Image 评估脚本**
   - 对齐 scripts/evaluate_text_reid.py 的接口，使其能正确加载模型权重、构建文本和图像 dataloader，输出 Text→Image 检索的 R@1/5/10 + mAP

### 阶段 2：核心消融实验 — Market1501（Day 4-10，~40 GPU 小时）

所有实验基于 Market1501，使用统一测试配置 `SIZE_TEST=[384,128], IMS_PER_BATCH=256`，seed=1234。

5. **实验 B0（对照基线）：重新训练当前 Text-Guided 管线**
   - 配置：`TEXT_ENCODER_TYPE=clip_native`，Stage1=120ep + Stage2=60ep，batch=32
   - 目标指标：复现 88.0% mAP / 94.3% R-1
   - 预计耗时：~6 小时

6. **实验 B1：TextQueryEncoder 替换**
   - 配置：`TEXT_ENCODER_TYPE=query_encoder`，其余同 B0
   - 预期：属性注意力机制应提升文本特征质量，预估 +0.5~1.5% mAP
   - 预计耗时：~6 小时

7. **实验 B2：Stage 2 加入 CrossModalContrastiveLoss**
   - 配置：B0 基础上 Stage 2 额外启用 `CrossModalContrastiveLoss`（`TEXT_LOSS_TYPE=contrastive, TEXT_LOSS_WEIGHT=0.5`）
   - 预期：Stage 2 的持续图文对齐应减少表征漂移，+0.5~1.0% mAP
   - 预计耗时：~6 小时

8. **实验 B3：I2T 权重消融**
   - 配置：B0 基础上分别设置 `I2T_LOSS_WEIGHT` = {0.2, 0.5, 1.0, 2.0}
   - 预期：找到最优 I2T 权重平衡点
   - 预计耗时：~24 小时（4 组 × 6 小时）

9. **实验 B4：最优组合**
   - 把 B1~B3 中的最优配置组合：`TextQueryEncoder` + 最优 `I2T_LOSS_WEIGHT` + Stage2 跨模态损失
   - 预期：达到或超过 SIE+OLP 基线的 89.6% mAP
   - 预计耗时：~6 小时

9b. **实验 B5：B3a 最优配置 + SIE + OLP** ✅
   - 配置：`I2T_LOSS_WEIGHT=0.2` + `SIE_CAMERA=True` + `SIE_COE=3.0` + `STRIDE_SIZE=[12,12]`，完整 Stage1+Stage2 训练
   - 动机：Phase 2 实验均未开启 SIE/OLP，原始 CLIP-ReID 基线的 89.6% mAP 包含 SIE+OLP，需验证 Text-Guided + SIE+OLP 的叠加效果
   - **实际结果：mAP 88.8%, Rank-1 94.1%**（+1.6% mAP vs B3a，接近基线 89.6%）
   - 实际耗时：~2.5 小时（RTX 5090）
   - Checkpoint: `/root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B5_i2t_w0.2_sie_olp/ViT-B-16_stage2_60.pth`

### 阶段 3：BLIP 标注生成与 MSMT17 验证（Day 11-16，~10h CPU + 30h GPU）

10. **配置 BLIP-2 并为 MSMT17 生成文本标注**
    - 使用已有脚本 scripts/generate_captions_blip.py 为 MSMT17 的 ~32K 训练图像生成描述
    - 下载 `Salesforce/blip2-opt-2.7b`（~6GB），需 GPU 推理 ~4 小时
    - 替换 annotations/msmt17_train.json 中的占位文本
    - 同时为 Market1501 也生成 BLIP 标注，用于消融对比（属性模板 vs BLIP 自然语言）

11. **实验 C1：MSMT17 Text-Guided（BLIP 标注）** ✅
    - 用阶段 2 的最优配置（I2T=0.2, no SIE, stride [16,16]）训练 MSMT17
    - **实际结果：mAP 68.1%, Rank-1 85.2%**（低于目标 72.0%，基线 73.4%）
    - 分析：BLIP-2 对低分辨率监控图质量有限，文本截断损失语义，未加 SIE+OLP
    - 实际耗时：~2.8 小时（RTX 5090，Stage1 82min + Stage2 90min）
    - Checkpoint: `/root/autodl-tmp/CLIP_REID/OUTPUT/phase3/C1_msmt17_blip/ViT-B-16_stage2_60.pth`

12. **实验 C2：Market1501 文本来源消融** 🔄
    - 对比 Market1501 上属性模板文本（B3a）vs BLIP-2 生成文本的效果
    - 配置：I2T=0.2, no SIE, stride [16,16]，annotations/market1501_train_blip.json
    - 对照组：B3a（属性模板, mAP 87.2%, Rank-1 93.9%）
    - 状态：运行中

### 阶段 4：跨模态损失消融与深层分析（Day 17-21，~20 GPU 小时）

13. **实验 D1：跨模态损失类型消融**
    - 在 B4 最优基础上，分别用 `contrastive` / `triplet` / `cmpm` / `combined` 四种跨模态损失训练
    - 预期：确定单损失最优和组合收益
    - 预计耗时：~24 小时（4 组 × 6 小时）

14. **特征空间可视化分析**
    - 对 B0（对照）和 B4（最优组合）提取测试集特征
    - t-SNE 可视化：文本特征与图像特征的对齐程度
    - 相似度分布直方图：正样本对 vs 负样本对的余弦相似度分布
    - 检索可视化：用 scripts/evaluate_text_reid.py 生成 top-k 检索结果对比

15. **多 seed 可靠性验证**
    - 对最终最优配置跑 seed={1234, 777, 4321}，报告均值±标准差
    - 预计耗时：~18 小时（3 seed × 6h）

---

**实验总览矩阵**

| 编号 | 实验 | 变量 | 阶段 | 预计 GPU 时间 |
|------|------|------|------|-------------|
| B0 | Text-Guided 对照 (clip_native) | — | 2 | 6h |
| B1 | + TextQueryEncoder | TEXT_ENCODER_TYPE | 2 | 6h |
| B2 | + Stage2 CrossModal Loss | TEXT_LOSS_TYPE | 2 | 6h |
| B3 | I2T 权重消融 ×4 | I2T_LOSS_WEIGHT | 2 | 24h |
| B4 | 最优组合 | B1+B2+B3 best | 2 | 6h |
| B5 | B3a最优 + SIE + OLP | SIE_CAMERA+STRIDE_SIZE | 2 | 3h |
| C1 | MSMT17 + BLIP 标注 | dataset + annotation | 3 | 10h |
| C2 | Market1501 文本来源消融 | attribute vs BLIP | 3 | 6h |
| D1 | 跨模态损失类型消融 ×4 | loss type | 4 | 24h |
| D2 | 多 seed 验证 ×3 | seed | 4 | 18h |

**总计：~109 GPU 小时**（3090 约 ￥300-400 AutoDL 费用，RTX 5090 实际更快）

---

**Verification**

- **阶段 1 验证**：代码修改后用 `python train_text_reid.py --config-file configs/person/vit_clipreid_market1501.yml` 干跑 2 epoch，确认 `TextQueryEncoder` 前向正常、损失计算无报错
- **每组实验验证**：检查 test log 中 mAP / Rank-1 指标；对比前一实验确认单变量效果方向
- **阶段 2 实际结果（RTX 5090, 2026-02-10）**：
  | 实验 | 配置 | mAP | Rank-1 | 状态 |
  |------|------|-----|--------|------|
  | B0 | 对照基线 (clip_native, I2T=1.0) | 84.2% | 92.8% | ✅ |
  | B1 | TextQueryEncoder | 86.5% | 93.9% | ✅ |
  | B2 | +CrossModal contrastive (w=0.5) | 83.8% | 93.2% | ✅ |
  | B3a | I2T=0.2 | **87.2%** | **93.9%** | ✅ 最优 |
  | B3b | I2T=0.5 | 86.8% | 93.8% | ✅ |
  | B3c | I2T=1.0 | 84.4% | 93.1% | ✅ |
  | B3d | I2T=2.0 | 80.0% | 91.6% | ✅ |
  | B5 | B3a + SIE + OLP | **88.8%** | **94.1%** | ✅ 全局最优 |
- **关键发现**：I2T 权重越低效果越好（0.2 > 0.5 > 1.0 > 2.0）；CrossModal Loss 无效；TextQueryEncoder 有正向效果；B5 叠加 SIE+OLP 后达到 88.8% mAP（+1.6% vs B3a），接近原始基线 89.6%
- **阶段 3 实际结果（RTX 5090, 2026-02-11）**：
  | 实验 | 配置 | mAP | Rank-1 | 状态 |
  |------|------|-----|--------|------|
  | C1 | MSMT17 + BLIP-2 标注 (I2T=0.2, no SIE) | 68.1% | 85.2% | ✅ |
  | C2 | Market1501 + BLIP-2 标注 (I2T=0.2, no SIE) | — | — | 🔄 运行中 |
- **C1 分析**：MSMT17 BLIP 标注 68.1% mAP，低于目标 72.0%（基线 73.4%）。可能原因：(1) BLIP-2 对监控低分辨率图像的描述质量有限，存在重复/多人混淆；(2) 未加 SIE+OLP；(3) 文本截断损失部分语义信息。后续可尝试 C1 + SIE+OLP 组合。
- **阶段 3 验证**：BLIP-2 Q&A 格式生成 MSMT17（32621条）和 Market1501（12936条）标注，质量经抽检良好
- **最终验证**：B5 (Text-Guided + SIE+OLP) Market1501 mAP=88.8%（接近基线 89.6%，差距 0.8%）；MSMT17 需进一步调优

---

**Decisions**

- **优先完善现有模块而非引入新架构**：`TextQueryEncoder` 和跨模态损失已经实现但未接入，投入产出比最高
- **Market1501 先行 → MSMT17 跟进**：Market1501 有高质量属性标注可直接验证机制有效性，MSMT17 需先解决标注问题
- **阶段 2 的 I2T 权重消融选 4 个点**：{0.2, 0.5, 1.0, 2.0} 覆盖了从弱监督到强监督的范围，成本可控
- **跨模态损失消融放在阶段 4**：先通过阶段 2 确定编码器和权重的最优配置，再叠加不同损失类型，避免搜索空间爆炸
- **新增 B5 实验验证 SIE+OLP 叠加效果**：Phase 2 结果显示所有实验均未开启 SIE/OLP，而原始基线 89.6% 包含 SIE+OLP，因此在最优的 B3a (I2T=0.2, 87.2%) 基础上叠加 SIE+OLP，预期可达 89%+

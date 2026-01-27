# Text-Guided ReID 文本接入补丁说明

**目标**：把 JSON 文本真正接入训练前向（Stage‑1/Stage‑2），让 I2T 对齐使用真实文本。

## 变更点
1) **模型层：支持直接编码 text tokens**
- 文件：`model/make_model_clipreid.py`
- 新增：`encode_text_tokens(text_tokens)` + `get_text_tokens` 分支
- 行为：允许外部传入 `clip.tokenize` 生成的 `text_tokens`，直接走 CLIP 文本编码路径

2) **数据层：默认使用 CLIP tokenizer**
- 文件：`datasets/text_reid_dataset.py`
- 修改：`make_text_dataloader()` 默认 `tokenizer = clip.tokenize`
- Collator 会产出 `text_tokens`（shape: [B,77]）

3) **训练层：Stage‑1 使用真实文本**
- 文件：`processor/processor_text_guided.py`
- 修改：若 batch 含 `text_tokens`，使用 `model(get_text_tokens=True, text_tokens=...)`
- 若缺失，则回退 prompt 文本（兼容旧数据）

4) **训练层：Stage‑2 加入 I2T Loss**
- 文件：`processor/processor_text_guided.py`
- 修改：生成 `i2t_logits = image_features @ text_features.T`，传入 `make_loss`

## 使用说明
- 只要 `annotations/*.json` 含 `text` 字段，即可生效
- 兼容旧流程：若 `text_tokens` 不存在，自动回退到 prompt 文本


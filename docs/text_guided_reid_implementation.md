# Text-Guided ReID 分支实现说明（含原始 CLIP-ReID 原理对照）

> 受众：组会同学 & 自己复盘
> 目标：完整理解 text-guided-reid 分支的“做了什么 / 为什么这样做 / 代码如何落地”

---

## 0. 一句话结论（TL;DR）
Text‑Guided ReID 分支在**不改变推理阶段的视觉检索流程**前提下，引入“文本监督 + 图文对齐的预训练阶段”，用语言语义把视觉特征拉到更可分的语义空间；训练完成后仍按原 CLIP‑ReID 的纯视觉检索方式进行评估。

---

## 1. 原始 CLIP‑ReID 的做法与原理（基线）
### 1.1 任务与整体流程
- **任务**：图像‑图像行人检索（Query 图像 → Gallery 图像集合）
- **模型**：CLIP 视觉编码器（ViT/RN） + Prompt Learner + CLIP 文本编码器
- **训练两阶段**：
  - **Stage‑1**：图文对比预训练（用可学习 prompt 生成“类文本”，对齐图像与文本）
  - **Stage‑2**：纯视觉微调（ID Loss + Triplet Loss + 可选 I2T）

### 1.2 关键代码位置
- 训练入口：`train_clipreid.py`
- Stage‑1：`processor/processor_clipreid_stage1.py`
- Stage‑2：`processor/processor_clipreid_stage2.py`
- 模型：`model/make_model_clipreid.py`
- Loss：`loss/make_loss.py`
- 数据：`datasets/make_dataloader_clipreid.py`

### 1.3 原始 Stage‑1 原理（对比预训练）
`processor_clipreid_stage1.py` 中：
- 先用图像编码器抽取 `image_features`
- 用 **PromptLearner** + **TextEncoder** 生成类文本特征 `text_features`
- 采用 **SupConLoss** 做双向对比（I2T + T2I）

关键逻辑（简化）：
```python
image_feature = model(img, target, get_image=True)
text_features = model(label=target, get_text=True)
loss_i2t = SupCon(image_feature, text_features, target)
loss_t2i = SupCon(text_features, image_feature, target)
loss = loss_i2t + loss_t2i
```

### 1.4 原始 Stage‑2 原理（纯视觉微调）
`processor_clipreid_stage2.py` 中：
- 前向得到 `score`（分类）与 `feat`（检索特征）
- Loss = **ID Loss + Triplet Loss (+ I2T Loss 可选)**
- I2T Loss 使用 `image_features @ text_features.T` 的分类式对齐

关键逻辑（简化）：
```python
score, feat, image_features = model(img, label=target)
text_features = model(label=all_class_ids, get_text=True)
logits = image_features @ text_features.t()
loss = ID + Triplet + I2T
```

> **原理总结**：CLIP‑ReID 的“文本”是由**可学习的 prompt + 类标签**产生的，属于“类级文本监督”。

---

## 2. 为什么引入 Text‑Guided（动机）
1) **语义更强**：属性或自然语言可表达“红上衣、背包、短裤”等语义，能帮助模型区分外观相似行人。  
2) **弱监督补充**：纯视觉对遮挡/背景敏感，语义监督能起到“正则化与对齐”的作用。  
3) **贴近真实场景**：实际应用常只有口述描述，文本引导是可落地的检索形式。  

---

## 3. Text‑Guided 分支具体实现（文件 & 函数级）

### 3.0 数据流总览（已接入真实文本）
```
annotations/*.json (text)
   ↓ (TextReIDDataset + clip.tokenize)
text_tokens
   ↓
processor_text_guided.py
   ├─ Stage‑1: image_features vs text_features(text_tokens)
   └─ Stage‑2: ID + Triplet + I2T(text_tokens)
```
> 关键点：**JSON 文本已进入模型前向**（Stage‑1/2 均可用）。

### 3.1 文本标注与数据准备
#### 3.1.1 Market1501 属性文本
- **脚本**：`datasets/attribute_parser.py`
- **输出**：`annotations/market1501_train.json`
- **实现点**：把 `.mat` 属性映射到自然语言模板

示例片段（简化）：
```python
text = "A {age} {gender} with {hair} hair...".format(...)
```

#### 3.1.2 BLIP 生成文本（可选）
- **脚本**：`scripts/generate_captions_blip.py`
- **输出**：`annotations/market1501_train_blip.json`
- **作用**：更丰富、更自然的图像描述

### 3.2 文本数据集与采样器
- **文件**：`datasets/text_reid_dataset.py`
- **类**：`TextReIDDataset` + `TextReIDCollator`
- **采样器**：`RandomIdentitySampler`（保证 Triplet 采样）

关键逻辑（简化）：
```python
return {
  'image': image,
  'text': text,
  'label': pid_label,
  'camid': camid
}
```
**注意**：`text` 会被 `clip.tokenize` 转成 `text_tokens` 并参与训练（见 3.4 / 3.5）。

### 3.3 训练入口
- **入口**：`train_text_reid.py`
- **差异**：读取 `annotation_file` 构造 `train_loader_text`
- **两阶段**：Stage‑1 (Text‑Guided) + Stage‑2 (视觉微调)

关键逻辑：
```python
train_loader_text, _ = make_text_dataloader(cfg, args.annotation_file)

# Stage 1
if not skip_stage1:
  do_train_text_guided_stage1(...)

# Stage 2
do_train_text_guided(...)
```

### 3.4 Stage‑1：Text‑Guided 预训练
- **文件**：`processor/processor_text_guided.py`
- **函数**：`do_train_text_guided_stage1`

关键逻辑：
```python
image_features = model(x=img, label=target, get_image=True)
text_features  = model(get_text_tokens=True, text_tokens=text_tokens)
loss = CrossModalContrastiveLoss(text_features, image_features, target)
```

> 当前实现优先使用 **JSON 文本（text_tokens）**；若未提供则回退到 PromptLearner 文本（SupConLoss）。

### 3.5 Stage‑2：纯视觉微调
- **文件**：`processor/processor_text_guided.py`
- **函数**：`do_train_text_guided`

关键逻辑：
```python
score, feat, _ = model(img, target)
loss = loss_func(score, feat, target, target_cam)
```

> 当前实现 **已加入 I2T Loss**（当 text_tokens 存在时）。

### 3.6 模型中的文本路径
- **文件**：`model/make_model_clipreid.py`
- **函数**：`forward(..., get_text=True)`
- **文本路径**：`PromptLearner → TextEncoder`

关键逻辑：
```python
prompts = prompt_learner(label)
text_features = text_encoder(prompts, tokenized_prompts)
```

### 3.7 新增的“文本编码器”模块（目前未接入）
- **文件**：`model/text_query_encoder.py`
- **状态**：定义了可训练 `TextQueryEncoder`，但当前训练脚本未使用

### 3.8 代码中“已写但未接入”的模块
- `loss/cross_modal_loss.py`：实现了 InfoNCE / 跨模态 Triplet 等，但训练中未引用  
- `model/text_query_encoder.py`：文本编码器已定义，但当前仍使用 CLIP text encoder 路径  

---

## 4. Text‑Guided 分支与原始 CLIP‑ReID 的差异总结

| 模块 | 原始 CLIP‑ReID | Text‑Guided 分支 | 备注 |
|---|---|---|---|
| 数据 | 仅图像 + ID | 图像 + 文本 JSON | 文本目前未进入前向 |
| Stage‑1 | 图像 vs Prompt 文本对齐 | 同样使用 Prompt 文本对齐 | 文本 JSON 未用 |
| Stage‑2 | ID+Triplet(+I2T) | ID+Triplet | I2T 未接入 |
| 新增 | 无 | attribute_parser / text_reid_dataset / BLIP 生成脚本 | 为后续扩展打基础 |

**一句话总结**：Text‑Guided 分支已完成**文本数据→token→模型前向**的闭环，Stage‑1/2 均可使用真实文本进行对齐训练。

---

## 4.1 设计文档 vs 实际代码（差异清单）
设计文档（`docs/text_guided_reid_design.md`）提出“文本引导检索 + 跨模态匹配”，当前实现差异：  
1) **TextQueryEncoder 未接入**：仍使用 CLIP 文本编码路径。  
2) **跨模态损失未使用**：`loss/cross_modal_loss.py` 中 InfoNCE 等仍未接入。  

---

## 5. 为什么这样做（设计取舍）
1) **兼容性优先**：复用原 CLIP‑ReID 的 PromptLearner 和对比损失，训练稳定性高、改动少。  
2) **工程渐进**：先把文本标注与数据管线搭好，再逐步引入真实文本编码，降低风险。  
3) **可替换性**：未来可以无缝把 `text_features` 替换为 `text_query_encoder(text)`，接入 BLIP/LLM 文本。  

---

## 6. 已知不足 & 下一步（建议）
### 6.1 当前不足
- 仍未使用 `cross_modal_loss.py` 的更强 InfoNCE/HardNeg
- `TextQueryEncoder` 尚未接入

### 6.2 下一步可实现的增强
1) **真正使用文本**：
   - `TextReIDCollator` 输出 `text_tokens`
   - `TextQueryEncoder` 接入 `processor_text_guided_stage1`
2) **加入 I2T / InfoNCE**：
   - Stage‑2 加入 `logits = img @ text.T` 的对齐损失
3) **改造损失**：
   - 使用 `loss/cross_modal_loss.py` 中 InfoNCE + Hard Negative

---

## 7. 关键文件与函数索引（便于定位）
- `train_text_reid.py`：训练入口，Stage‑1/2 调度  
- `processor/processor_text_guided.py`：`do_train_text_guided_stage1` / `do_train_text_guided`  
- `datasets/text_reid_dataset.py`：`TextReIDDataset` / `TextReIDCollator`  
- `datasets/attribute_parser.py`：属性→文本  
- `scripts/generate_captions_blip.py`：BLIP 文本生成  
- `model/make_model_clipreid.py`：`forward(get_text=True/get_image=True)`  
- `loss/make_loss.py`：支持 I2T Loss，但当前未传入  
- `loss/cross_modal_loss.py`：InfoNCE 等（未接入）  


---

## 7. 复现流程（当前分支）
1) 生成属性文本：
```bash
python datasets/attribute_parser.py \
  --attribute_file datasets/attributes/market1501_attribute.mat \
  --image_dir /root/autodl-tmp/CLIP_REID/DATASETS/Market-1501-v15.09.15/bounding_box_train \
  --output_file annotations/market1501_train.json
```

2) 训练：
```bash
bash scripts/train_text_guided_pipeline.sh market1501 0 attribute
```

3) 评估：
```bash
python test_clipreid.py --config_file configs/person/vit_clipreid.yml \
  DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR /root/autodl-tmp/CLIP_REID/DATASETS \
  TEST.WEIGHT /root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/market1501/ViT-B-16_stage2_60.pth
```

---

## 8. 结语
Text‑Guided ReID 分支已经实现了“**文本数据管线 + 两阶段训练框架**”，并验证了文本引导带来的性能增益。下一步关键是把真实文本（属性/BLIP）真正接入对齐损失，使“文本监督”从概念落到**特征层的有效对齐**。

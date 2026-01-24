# Text-Guided Person Re-Identification

## 研究方向概述

**核心思想**：利用 CLIP 的跨模态能力，让用户通过自然语言描述来检索监控视频中的行人，而非传统的图像-图像匹配。

**创新点**：
1. 定义新任务：Text-to-Image Person ReID
2. 提出 Prompt-based Query Generation
3. 设计 Cross-modal Alignment Loss

---

## 问题定义

### 传统 ReID
```
Query: 图像 (probe image)
Gallery: 图像集合
目标: 找到与 Query 相同身份的图像
```

### Text-Guided ReID (Ours)
```
Query: 自然语言描述 (e.g., "穿红色外套、戴眼镜的男性")
Gallery: 图像集合
目标: 找到与文本描述匹配的行人图像
```

---

## 技术方案

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Text-Guided ReID                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Text Query                        Image Gallery            │
│  "A man in red                     ┌───┐ ┌───┐ ┌───┐       │
│   jacket with                      │img│ │img│ │img│ ...   │
│   glasses"                         └───┘ └───┘ └───┘       │
│       │                                  │                  │
│       ▼                                  ▼                  │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ CLIP Text    │              │ CLIP Vision  │            │
│  │ Encoder      │              │ Encoder      │            │
│  └──────────────┘              └──────────────┘            │
│       │                                  │                  │
│       ▼                                  ▼                  │
│  Text Feature                      Image Features          │
│  [1, 512]                          [N, 512]                │
│       │                                  │                  │
│       └──────────────┬───────────────────┘                  │
│                      ▼                                      │
│              Similarity Matrix                              │
│              [1, N]                                         │
│                      │                                      │
│                      ▼                                      │
│              Ranked Results                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2. 关键模块

#### 2.1 Text Encoder (基于 CLIP)

```python
class TextQueryEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # 可学习的查询增强模块
        self.query_refiner = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
    
    def forward(self, text):
        # CLIP text encoding
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        text_feat = x @ self.text_projection
        
        # Query refinement
        text_feat = self.query_refiner(text_feat) + text_feat
        return text_feat
```

#### 2.2 Cross-modal Matching

```python
def text_to_image_matching(text_features, image_features, temperature=0.07):
    """
    Args:
        text_features: [1, 512] 或 [B, 512]
        image_features: [N, 512] gallery 中所有图像的特征
    Returns:
        similarity: [1, N] 或 [B, N]
    """
    # L2 归一化
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    
    # 余弦相似度
    similarity = text_features @ image_features.T / temperature
    
    return similarity
```

#### 2.3 训练损失

**Cross-modal Contrastive Loss**:
```python
class TextImageContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, text_features, image_features, labels):
        # 计算相似度
        logits = text_features @ image_features.T / self.temperature
        
        # 创建正样本 mask
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()
        
        # InfoNCE loss
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # 只计算正样本
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        return loss.mean()
```

---

## 数据集需求

### 现有数据集扩展

需要为现有 ReID 数据集添加文本标注：

| 数据集 | 图像数 | 身份数 | 标注策略 |
|--------|--------|--------|----------|
| Market-1501 | 32,668 | 1,501 | 基于属性自动生成 |
| MSMT17 | 126,441 | 4,101 | 基于属性自动生成 |
| CUHK-PEDES | 40,206 | 13,003 | 已有人工标注 ✓ |

### 文本标注生成策略

**方案 A：基于属性模板**
```python
# 属性 → 文本模板
attributes = {
    "gender": "male",
    "age": "young",
    "upper_color": "red",
    "lower_color": "black",
    "bags": "backpack"
}

def generate_description(attrs):
    parts = []
    if attrs["gender"]:
        parts.append(f"A {attrs['age']} {attrs['gender']}")
    if attrs["upper_color"]:
        parts.append(f"wearing {attrs['upper_color']} top")
    if attrs["lower_color"]:
        parts.append(f"and {attrs['lower_color']} pants")
    if attrs["bags"]:
        parts.append(f"carrying a {attrs['bags']}")
    return " ".join(parts) + "."

# 输出: "A young male wearing red top and black pants carrying a backpack."
```

**方案 B：使用 VLM 生成**
```python
# 使用 BLIP-2 / LLaVA 生成描述
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def generate_caption(image):
    inputs = processor(image, "Describe this person's appearance", 
                       return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0])
```

---

## 实验计划

### Phase 1: 基础验证 (Week 1-2)
- [ ] 在 CUHK-PEDES 数据集上验证基础方案
- [ ] 实现 TextQueryEncoder
- [ ] 实现基础的 text-to-image 匹配

### Phase 2: 方法改进 (Week 3-4)
- [ ] 设计 Query Refinement 模块
- [ ] 实现 Cross-modal Contrastive Loss
- [ ] 添加属性级别的对齐损失

### Phase 3: 扩展实验 (Week 5-6)
- [ ] 为 Market-1501 生成文本标注
- [ ] 跨数据集泛化实验
- [ ] 消融实验

### Phase 4: 论文撰写 (Week 7-8)
- [ ] 整理实验结果
- [ ] 撰写论文
- [ ] 投稿目标：CVPR / ICCV / ECCV / AAAI

---

## 评估指标

| 指标 | 描述 |
|------|------|
| Rank-1 | 第一个返回结果是否正确 |
| Rank-5 | 前5个结果中是否有正确答案 |
| Rank-10 | 前10个结果中是否有正确答案 |
| mAP | 平均精度均值 |

---

## 参考文献

1. **CLIP-ReID**: Li et al., "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification", AAAI 2024
2. **CUHK-PEDES**: Li et al., "Person Search with Natural Language Description", CVPR 2017
3. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

---

## 分支信息

- **分支名**: `exp/text-guided-reid`
- **基于**: `master`
- **创建时间**: 2026-01-24

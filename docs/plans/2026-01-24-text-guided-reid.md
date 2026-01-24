# Text-Guided ReID Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 Market-1501 和 MSMT17 数据集生成文本描述，并实现基于文本查询的行人重识别系统。

**Architecture:** 
- 使用两种方案生成文本标注：(A) 属性模板生成 (B) BLIP 自动生成
- 修改 CLIP-ReID 支持 text-to-image 检索模式
- 新增 TextQueryEncoder 和 CrossModalMatcher 模块

**Tech Stack:** PyTorch, CLIP, BLIP-2 (transformers), Market-1501 Attribute

---

## Task 1: 下载 Market-1501 属性标注

**Files:**
- Create: `datasets/attributes/market1501_attribute.mat`
- Create: `scripts/download_attributes.sh`

**Step 1: 创建下载脚本**

```bash
#!/bin/bash
# scripts/download_attributes.sh
# 下载 Market-1501 属性标注文件

mkdir -p datasets/attributes
cd datasets/attributes

# 从官方仓库下载
wget https://github.com/vana77/Market-1501_Attribute/raw/master/market_attribute.mat -O market1501_attribute.mat

echo "Market-1501 属性文件下载完成!"
```

**Step 2: 执行下载**

Run: `bash scripts/download_attributes.sh`
Expected: 文件 `datasets/attributes/market1501_attribute.mat` 存在

**Step 3: Commit**

```bash
git add scripts/download_attributes.sh
git commit -m "chore: add script to download market1501 attributes"
```

---

## Task 2: 创建属性解析器

**Files:**
- Create: `datasets/attribute_parser.py`
- Test: `tests/test_attribute_parser.py`

**Step 1: 创建属性解析器**

```python
# datasets/attribute_parser.py
"""
解析 Market-1501 属性标注文件，将属性转换为文本描述
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

# Market-1501 属性定义
MARKET_ATTRIBUTES = {
    'gender': ['female', 'male'],
    'hair': ['short hair', 'long hair'],
    'sleeve': ['long sleeve', 'short sleeve'],
    'length': ['long lower body clothing', 'short lower body clothing'],
    'type': ['wearing dress', 'wearing pants'],
    'hat': ['no hat', 'wearing hat'],
    'backpack': ['no backpack', 'carrying backpack'],
    'bag': ['no handbag', 'carrying handbag'],
    'handbag': ['no bag', 'carrying bag'],
    'age': ['young', 'teenager', 'adult', 'old'],
    'upcolor': ['black', 'white', 'red', 'purple', 'yellow', 'gray', 'blue', 'green'],
    'downcolor': ['black', 'white', 'pink', 'purple', 'yellow', 'gray', 'blue', 'green', 'brown'],
}

class Market1501AttributeParser:
    """解析 Market-1501 属性标注"""
    
    def __init__(self, attribute_file: str):
        """
        Args:
            attribute_file: market1501_attribute.mat 文件路径
        """
        self.attribute_file = Path(attribute_file)
        self.attributes = None
        self.image_names = None
        self._load_attributes()
    
    def _load_attributes(self):
        """加载属性文件"""
        if not self.attribute_file.exists():
            raise FileNotFoundError(f"属性文件不存在: {self.attribute_file}")
        
        mat = sio.loadmat(str(self.attribute_file))
        # mat 结构: market_attribute -> train/test -> image_index, attributes
        self.market_attribute = mat['market_attribute'][0, 0]
    
    def get_attributes_by_pid(self, pid: int, split: str = 'train') -> dict:
        """
        获取指定身份的属性
        
        Args:
            pid: 行人身份ID
            split: 'train' 或 'test'
        
        Returns:
            dict: 属性字典
        """
        # 实现细节取决于 mat 文件的具体结构
        pass
    
    def attributes_to_text(self, attrs: dict) -> str:
        """
        将属性字典转换为自然语言描述
        
        Args:
            attrs: 属性字典，如 {'gender': 1, 'upcolor': 2, ...}
        
        Returns:
            str: 自然语言描述
        """
        parts = []
        
        # 年龄和性别
        gender = MARKET_ATTRIBUTES['gender'][attrs.get('gender', 1)]
        age = MARKET_ATTRIBUTES['age'][attrs.get('age', 2)]
        parts.append(f"A {age} {gender}")
        
        # 上衣
        upcolor = MARKET_ATTRIBUTES['upcolor'][attrs.get('upcolor', 0)]
        sleeve = MARKET_ATTRIBUTES['sleeve'][attrs.get('sleeve', 0)]
        parts.append(f"wearing a {upcolor} {sleeve} top")
        
        # 下装
        downcolor = MARKET_ATTRIBUTES['downcolor'][attrs.get('downcolor', 0)]
        lower_type = "pants" if attrs.get('type', 1) == 1 else "dress"
        parts.append(f"and {downcolor} {lower_type}")
        
        # 配件
        accessories = []
        if attrs.get('hat', 0) == 1:
            accessories.append("a hat")
        if attrs.get('backpack', 0) == 1:
            accessories.append("a backpack")
        if attrs.get('bag', 0) == 1:
            accessories.append("a bag")
        
        if accessories:
            parts.append("with " + " and ".join(accessories))
        
        return " ".join(parts) + "."


def generate_text_annotations(
    attribute_file: str,
    image_dir: str,
    output_file: str
):
    """
    为所有图像生成文本标注
    
    Args:
        attribute_file: 属性标注文件路径
        image_dir: 图像目录路径
        output_file: 输出的 JSON 文件路径
    """
    import json
    from glob import glob
    
    parser = Market1501AttributeParser(attribute_file)
    
    annotations = {}
    image_paths = glob(f"{image_dir}/*.jpg")
    
    for img_path in image_paths:
        img_name = Path(img_path).name
        # 解析 PID: 0001_c1s1_000151_01.jpg -> 0001
        pid = int(img_name.split('_')[0])
        
        try:
            attrs = parser.get_attributes_by_pid(pid)
            text = parser.attributes_to_text(attrs)
            annotations[img_name] = {
                'pid': pid,
                'text': text,
                'attributes': attrs
            }
        except Exception as e:
            print(f"Warning: 无法处理 {img_name}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"生成了 {len(annotations)} 条文本标注，保存到 {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute_file', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()
    
    generate_text_annotations(args.attribute_file, args.image_dir, args.output_file)
```

**Step 2: Commit**

```bash
git add datasets/attribute_parser.py
git commit -m "feat: add Market-1501 attribute parser for text generation"
```

---

## Task 3: 创建 BLIP 文本生成器

**Files:**
- Create: `scripts/generate_captions_blip.py`

**Step 1: 创建 BLIP 生成脚本**

```python
# scripts/generate_captions_blip.py
"""
使用 BLIP-2 为 ReID 数据集生成文本描述

Usage:
    python scripts/generate_captions_blip.py \
        --image_dir /path/to/images \
        --output_file annotations.json \
        --batch_size 16
"""

import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class ImageDataset(Dataset):
    """简单的图像数据集"""
    
    def __init__(self, image_dir: str, processor):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        self.processor = processor
        
        # 过滤掉无效图像（如 -1 开头的 junk 图像）
        self.image_paths = [p for p in self.image_paths if not p.name.startswith('-1')]
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (128, 256), color='white')
        
        return {
            'image': image,
            'path': str(img_path),
            'name': img_path.name
        }


def generate_captions(
    image_dir: str,
    output_file: str,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    batch_size: int = 8,
    device: str = "cuda"
):
    """
    使用 BLIP-2 生成图像描述
    
    Args:
        image_dir: 图像目录
        output_file: 输出 JSON 文件
        model_name: BLIP-2 模型名称
        batch_size: 批次大小
        device: 设备
    """
    print(f"Loading BLIP-2 model: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    dataset = ImageDataset(image_dir, processor)
    
    # 自定义 collate_fn
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        paths = [item['path'] for item in batch]
        names = [item['name'] for item in batch]
        return images, paths, names
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 用于 ReID 任务的提示词
    prompt = "Describe this person's appearance including their clothing, accessories, and physical features."
    
    annotations = {}
    
    print("Generating captions...")
    for images, paths, names in tqdm(dataloader):
        # 处理输入
        inputs = processor(
            images=images, 
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True
        ).to(device, torch.float16)
        
        # 生成描述
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True
            )
        
        # 解码
        generated_texts = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        # 保存结果
        for name, path, text in zip(names, paths, generated_texts):
            # 解析 PID
            pid = int(name.split('_')[0]) if not name.startswith('-1') else -1
            
            annotations[name] = {
                'pid': pid,
                'text': text.strip(),
                'source': 'blip2'
            }
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(annotations)} captions, saved to {output_file}")
    
    # 打印一些示例
    print("\n=== Sample Captions ===")
    for i, (name, ann) in enumerate(list(annotations.items())[:5]):
        print(f"{name}: {ann['text']}")


def main():
    parser = argparse.ArgumentParser(description='Generate captions using BLIP-2')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--model_name', type=str, 
                        default="Salesforce/blip2-opt-2.7b",
                        help='BLIP-2 model name')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    generate_captions(
        image_dir=args.image_dir,
        output_file=args.output_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add scripts/generate_captions_blip.py
git commit -m "feat: add BLIP-2 caption generation script"
```

---

## Task 4: 创建文本标注数据集加载器

**Files:**
- Create: `datasets/text_reid_dataset.py`

**Step 1: 创建 TextReID 数据集类**

```python
# datasets/text_reid_dataset.py
"""
Text-Guided ReID 数据集加载器
支持 text-to-image 和 image-to-text 检索
"""

import json
import random
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class TextReIDDataset(Dataset):
    """
    Text-Guided ReID 数据集
    
    支持两种模式:
    1. text-to-image: 用文本查询图像
    2. image-to-text: 用图像查询文本（对比学习训练用）
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform=None,
        mode: str = 'train',  # 'train', 'query', 'gallery'
    ):
        """
        Args:
            image_dir: 图像目录路径
            annotation_file: 文本标注 JSON 文件路径
            transform: 图像变换
            mode: 数据集模式
        """
        self.image_dir = Path(image_dir)
        self.mode = mode
        
        # 加载标注
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 构建数据列表
        self.data = []
        for img_name, ann in self.annotations.items():
            img_path = self.image_dir / img_name
            if img_path.exists():
                self.data.append({
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'pid': ann['pid'],
                    'text': ann['text'],
                    'camid': self._parse_camid(img_name)
                })
        
        # 构建 PID 到索引的映射
        self.pid_to_indices = {}
        for idx, item in enumerate(self.data):
            pid = item['pid']
            if pid not in self.pid_to_indices:
                self.pid_to_indices[pid] = []
            self.pid_to_indices[pid].append(idx)
        
        # 图像变换
        if transform is None:
            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        print(f"TextReIDDataset [{mode}]: {len(self.data)} samples, "
              f"{len(self.pid_to_indices)} identities")
    
    def _parse_camid(self, img_name: str) -> int:
        """解析相机 ID，如 0001_c1s1_000151_01.jpg -> 1"""
        try:
            return int(img_name.split('_')[1][1])  # c1 -> 1
        except:
            return 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'text': item['text'],
            'pid': item['pid'],
            'camid': item['camid'],
            'image_path': item['image_path']
        }
    
    def get_text_by_pid(self, pid: int) -> List[str]:
        """获取指定身份的所有文本描述"""
        texts = []
        for idx in self.pid_to_indices.get(pid, []):
            texts.append(self.data[idx]['text'])
        return texts
    
    def get_all_pids(self) -> List[int]:
        """获取所有身份 ID"""
        return list(self.pid_to_indices.keys())


class TextReIDCollator:
    """
    自定义 collate_fn，处理文本和图像的批处理
    """
    
    def __init__(self, tokenizer=None, max_length: int = 77):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text'] for item in batch]
        pids = torch.tensor([item['pid'] for item in batch], dtype=torch.long)
        camids = torch.tensor([item['camid'] for item in batch], dtype=torch.long)
        
        # 如果有 tokenizer，则进行文本编码
        if self.tokenizer is not None:
            text_tokens = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            text_tokens = texts
        
        return {
            'images': images,
            'texts': texts,
            'text_tokens': text_tokens,
            'pids': pids,
            'camids': camids
        }


def build_text_reid_dataloaders(
    train_image_dir: str,
    train_annotation: str,
    query_image_dir: str,
    query_annotation: str,
    gallery_image_dir: str,
    gallery_annotation: str,
    batch_size: int = 64,
    num_workers: int = 4,
    tokenizer=None
):
    """
    构建 Text ReID 数据加载器
    
    Returns:
        train_loader, query_loader, gallery_loader
    """
    from torch.utils.data import DataLoader
    
    # 训练变换（带增强）
    train_transform = T.Compose([
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 测试变换
    test_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = TextReIDDataset(
        train_image_dir, train_annotation, train_transform, 'train'
    )
    query_dataset = TextReIDDataset(
        query_image_dir, query_annotation, test_transform, 'query'
    )
    gallery_dataset = TextReIDDataset(
        gallery_image_dir, gallery_annotation, test_transform, 'gallery'
    )
    
    collator = TextReIDCollator(tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, query_loader, gallery_loader
```

**Step 2: Commit**

```bash
git add datasets/text_reid_dataset.py
git commit -m "feat: add TextReID dataset loader with text annotations"
```

---

## Task 5: 创建 TextQueryEncoder 模块

**Files:**
- Create: `model/text_query_encoder.py`

**Step 1: 创建文本查询编码器**

```python
# model/text_query_encoder.py
"""
Text Query Encoder for Text-Guided ReID
基于 CLIP Text Encoder，添加可学习的查询优化模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TextQueryEncoder(nn.Module):
    """
    文本查询编码器
    
    将自然语言描述编码为特征向量，用于与图像特征匹配。
    基于 CLIP Text Encoder，添加了：
    1. Query Refinement: 优化查询表示
    2. Attribute Attention: 关注关键属性词
    """
    
    def __init__(
        self,
        clip_model,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_refinement_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            clip_model: CLIP 模型
            embed_dim: 输出特征维度
            hidden_dim: 隐藏层维度
            num_refinement_layers: Query Refinement 层数
            dropout: Dropout 比例
        """
        super().__init__()
        
        # CLIP 文本编码器组件
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        # Query Refinement 模块
        refinement_layers = []
        for _ in range(num_refinement_layers):
            refinement_layers.extend([
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout)
            ])
        self.query_refinement = nn.Sequential(*refinement_layers)
        
        # 属性注意力模块
        self.attribute_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 可学习的属性 queries
        # 关键属性: gender, age, upper_color, lower_color, bags, hat
        self.num_attributes = 6
        self.attribute_queries = nn.Parameter(
            torch.randn(self.num_attributes, embed_dim) * 0.02
        )
        
        # 输出投影
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.query_refinement.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.output_projection.weight, mode='fan_out')
        nn.init.constant_(self.output_projection.bias, 0)
    
    def encode_text(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        """
        编码文本（使用 CLIP 文本编码器）
        
        Args:
            tokenized_text: tokenized 文本，形状 [B, L]
        
        Returns:
            文本特征，形状 [B, D]
        """
        x = self.token_embedding(tokenized_text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 取 [EOS] token 的特征
        # [EOS] 是序列中最大的数字位置
        x = x[torch.arange(x.shape[0], device=x.device), 
              tokenized_text.argmax(dim=-1)] @ self.text_projection
        
        return x
    
    def forward(
        self,
        tokenized_text: torch.Tensor,
        return_all_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokenized_text: tokenized 文本，形状 [B, L]
            return_all_features: 是否返回所有中间特征
        
        Returns:
            优化后的查询特征，形状 [B, D]
        """
        # 1. CLIP 文本编码
        text_features = self.encode_text(tokenized_text)  # [B, D]
        
        # 2. Query Refinement
        refined_features = text_features + self.query_refinement(text_features)
        
        # 3. Attribute Attention
        # 使用可学习的属性 queries 来提取关键信息
        attr_queries = self.attribute_queries.unsqueeze(0).expand(
            text_features.shape[0], -1, -1
        )  # [B, num_attr, D]
        
        # 注意力计算
        attr_features, _ = self.attribute_attention(
            query=attr_queries,
            key=refined_features.unsqueeze(1),
            value=refined_features.unsqueeze(1)
        )  # [B, num_attr, D]
        
        # 平均池化属性特征
        attr_pooled = attr_features.mean(dim=1)  # [B, D]
        
        # 4. 融合
        fused_features = refined_features + 0.5 * attr_pooled
        
        # 5. 输出投影
        output_features = self.output_projection(fused_features)
        
        # L2 归一化
        output_features = F.normalize(output_features, p=2, dim=-1)
        
        if return_all_features:
            return {
                'text_features': text_features,
                'refined_features': refined_features,
                'attr_features': attr_features,
                'output_features': output_features
            }
        
        return output_features


class CrossModalMatcher(nn.Module):
    """
    跨模态匹配模块
    
    计算文本查询和图像特征之间的相似度
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * (1 / temperature))
        else:
            self.register_buffer('logit_scale', torch.ones([]) * (1 / temperature))
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        return_similarity: bool = True
    ) -> torch.Tensor:
        """
        计算文本与图像的匹配分数
        
        Args:
            text_features: 文本特征 [B_t, D] 或 [1, D]
            image_features: 图像特征 [B_i, D]
            return_similarity: 是否返回相似度矩阵
        
        Returns:
            相似度矩阵 [B_t, B_i]
        """
        # L2 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * text_features @ image_features.T
        
        return similarity


def build_text_query_encoder(clip_model, cfg=None):
    """
    构建 TextQueryEncoder
    
    Args:
        clip_model: CLIP 模型
        cfg: 配置对象
    
    Returns:
        TextQueryEncoder 实例
    """
    if cfg is not None:
        embed_dim = getattr(cfg.MODEL, 'TEXT_EMBED_DIM', 512)
        hidden_dim = getattr(cfg.MODEL, 'TEXT_HIDDEN_DIM', 512)
        num_layers = getattr(cfg.MODEL, 'TEXT_NUM_LAYERS', 2)
        dropout = getattr(cfg.MODEL, 'TEXT_DROPOUT', 0.1)
    else:
        embed_dim = 512
        hidden_dim = 512
        num_layers = 2
        dropout = 0.1
    
    return TextQueryEncoder(
        clip_model=clip_model,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_refinement_layers=num_layers,
        dropout=dropout
    )
```

**Step 2: Commit**

```bash
git add model/text_query_encoder.py
git commit -m "feat: add TextQueryEncoder with query refinement and attribute attention"
```

---

## Task 6: 创建跨模态损失函数

**Files:**
- Create: `loss/cross_modal_loss.py`

**Step 1: 创建损失函数**

```python
# loss/cross_modal_loss.py
"""
Cross-Modal Loss Functions for Text-Guided ReID

主要包含:
1. CrossModalContrastiveLoss: 跨模态对比学习损失
2. TextImageMatchingLoss: 文本-图像匹配损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """
    跨模态对比学习损失
    
    基于 InfoNCE，同时优化:
    - Text-to-Image: 给定文本，找到对应的图像
    - Image-to-Text: 给定图像，找到对应的文本
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        symmetric: bool = True
    ):
        """
        Args:
            temperature: 温度参数，控制分布的锐度
            label_smoothing: 标签平滑参数
            symmetric: 是否使用对称损失（T2I + I2T）
        """
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.symmetric = symmetric
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算跨模态对比损失
        
        Args:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]
            labels: 身份标签 [B]
        
        Returns:
            loss: 标量损失值
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # L2 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits = text_features @ image_features.T / self.temperature  # [B, B]
        
        # 构建正样本 mask（同一身份的为正样本）
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()  # [B, B]
        
        # 对角线也是正样本
        eye_mask = torch.eye(batch_size, device=device)
        pos_mask = torch.max(pos_mask, eye_mask)
        
        # Text-to-Image Loss
        t2i_logits_max, _ = logits.max(dim=1, keepdim=True)
        t2i_logits = logits - t2i_logits_max.detach()  # 数值稳定性
        t2i_exp = torch.exp(t2i_logits)
        t2i_log_prob = t2i_logits - torch.log(t2i_exp.sum(dim=1, keepdim=True))
        t2i_loss = -(pos_mask * t2i_log_prob).sum(dim=1) / pos_mask.sum(dim=1)
        t2i_loss = t2i_loss.mean()
        
        if self.symmetric:
            # Image-to-Text Loss
            i2t_logits = logits.T  # [B, B]
            i2t_logits_max, _ = i2t_logits.max(dim=1, keepdim=True)
            i2t_logits = i2t_logits - i2t_logits_max.detach()
            i2t_exp = torch.exp(i2t_logits)
            i2t_log_prob = i2t_logits - torch.log(i2t_exp.sum(dim=1, keepdim=True))
            i2t_loss = -(pos_mask.T * i2t_log_prob).sum(dim=1) / pos_mask.T.sum(dim=1)
            i2t_loss = i2t_loss.mean()
            
            loss = (t2i_loss + i2t_loss) / 2
        else:
            loss = t2i_loss
        
        return loss


class TextImageMatchingLoss(nn.Module):
    """
    文本-图像匹配损失
    
    结合多种损失:
    1. 跨模态对比损失
    2. 困难负样本挖掘
    3. 属性级别对齐（可选）
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.3,
        hard_negative_weight: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        
        self.contrastive_loss = CrossModalContrastiveLoss(
            temperature=temperature,
            symmetric=True
        )
    
    def _hard_negative_mining(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        困难负样本挖掘
        
        对于每个文本，找到最难区分的负样本图像
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # 计算相似度
        similarity = text_features @ image_features.T  # [B, B]
        
        # 构建 mask
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        neg_mask = 1 - pos_mask
        
        # 对于每个样本，找到最难的正样本和负样本
        # 最难正样本：相似度最低的正样本
        # 最难负样本：相似度最高的负样本
        
        # Mask 掉负样本，找最难正样本
        pos_sim = similarity * pos_mask + (-1e9) * neg_mask
        hardest_pos, _ = pos_sim.min(dim=1)  # [B]
        
        # Mask 掉正样本，找最难负样本
        neg_sim = similarity * neg_mask + (-1e9) * pos_mask
        hardest_neg, _ = neg_sim.max(dim=1)  # [B]
        
        # Triplet Margin Loss
        loss = F.relu(hardest_neg - hardest_pos + self.margin).mean()
        
        return loss
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        计算综合损失
        
        Args:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]
            labels: 身份标签 [B]
        
        Returns:
            dict: 包含各项损失和总损失
        """
        # 对比损失
        contrastive_loss = self.contrastive_loss(
            text_features, image_features, labels
        )
        
        # 困难负样本损失
        hard_neg_loss = self._hard_negative_mining(
            text_features, image_features, labels
        )
        
        # 总损失
        total_loss = contrastive_loss + self.hard_negative_weight * hard_neg_loss
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'hard_neg_loss': hard_neg_loss
        }


class CMPMLoss(nn.Module):
    """
    Cross-Modal Projection Matching (CMPM) Loss
    
    参考: Deep Cross-Modal Projection Learning (ECCV 2018)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 CMPM 损失
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # L2 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 构建标签矩阵
        labels = labels.view(-1, 1)
        label_mask = (labels == labels.T).float()  # [B, B]
        
        # 计算 KL 散度
        # P(i|t): 给定文本，图像的概率分布
        similarity = text_features @ image_features.T  # [B, B]
        
        # Softmax 归一化
        p_t2i = F.softmax(similarity, dim=1)  # [B, B]
        p_i2t = F.softmax(similarity.T, dim=1)  # [B, B]
        
        # 目标分布：同一身份的均匀分布
        target = label_mask / (label_mask.sum(dim=1, keepdim=True) + self.epsilon)
        
        # KL 散度
        kl_t2i = F.kl_div(
            torch.log(p_t2i + self.epsilon),
            target,
            reduction='batchmean'
        )
        kl_i2t = F.kl_div(
            torch.log(p_i2t + self.epsilon),
            target.T,
            reduction='batchmean'
        )
        
        return (kl_t2i + kl_i2t) / 2
```

**Step 2: Commit**

```bash
git add loss/cross_modal_loss.py
git commit -m "feat: add cross-modal loss functions for text-guided reid"
```

---

## Task 7: 创建评估脚本

**Files:**
- Create: `scripts/evaluate_text_reid.py`

**Step 1: 创建评估脚本**

```python
# scripts/evaluate_text_reid.py
"""
Text-Guided ReID 评估脚本

支持:
1. Text-to-Image Retrieval: 用文本查询图像
2. 计算 Rank-1, Rank-5, Rank-10, mAP 指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import json


def extract_image_features(
    model,
    dataloader,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    提取所有图像特征
    
    Returns:
        features: [N, D] 图像特征
        pids: [N] 身份 ID
        camids: [N] 相机 ID
    """
    model.eval()
    
    all_features = []
    all_pids = []
    all_camids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting image features'):
            images = batch['images'].to(device)
            pids = batch['pids'].numpy()
            camids = batch['camids'].numpy()
            
            # 提取特征
            features = model.encode_image(images)
            features = F.normalize(features, p=2, dim=-1)
            
            all_features.append(features.cpu())
            all_pids.extend(pids)
            all_camids.extend(camids)
    
    all_features = torch.cat(all_features, dim=0)
    all_pids = np.array(all_pids)
    all_camids = np.array(all_camids)
    
    return all_features, all_pids, all_camids


def extract_text_features(
    model,
    dataloader,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """
    提取所有文本特征
    
    Returns:
        features: [N, D] 文本特征
        pids: [N] 身份 ID
        texts: [N] 原始文本
    """
    model.eval()
    
    all_features = []
    all_pids = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting text features'):
            text_tokens = batch['text_tokens']
            if isinstance(text_tokens, dict):
                text_tokens = text_tokens['input_ids'].to(device)
            else:
                text_tokens = text_tokens.to(device)
            
            pids = batch['pids'].numpy()
            texts = batch['texts']
            
            # 提取特征
            features = model.encode_text(text_tokens)
            features = F.normalize(features, p=2, dim=-1)
            
            all_features.append(features.cpu())
            all_pids.extend(pids)
            all_texts.extend(texts)
    
    all_features = torch.cat(all_features, dim=0)
    all_pids = np.array(all_pids)
    
    return all_features, all_pids, all_texts


def compute_text_to_image_metrics(
    text_features: torch.Tensor,
    text_pids: np.ndarray,
    image_features: torch.Tensor,
    image_pids: np.ndarray,
    image_camids: np.ndarray,
    topk: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算 Text-to-Image 检索指标
    
    Args:
        text_features: 文本查询特征 [Q, D]
        text_pids: 文本查询身份 [Q]
        image_features: 图库图像特征 [G, D]
        image_pids: 图库图像身份 [G]
        image_camids: 图库图像相机 ID [G]
        topk: 计算 Rank-k 的 k 值列表
    
    Returns:
        metrics: 包含 Rank-k 和 mAP 的字典
    """
    num_query = text_features.shape[0]
    num_gallery = image_features.shape[0]
    
    # 计算相似度矩阵
    similarity = text_features @ image_features.T  # [Q, G]
    similarity = similarity.numpy()
    
    # 排序（降序，相似度高的排前面）
    indices = np.argsort(-similarity, axis=1)
    
    # 计算指标
    all_cmc = []
    all_ap = []
    
    for q_idx in range(num_query):
        q_pid = text_pids[q_idx]
        
        # 获取排序后的 gallery 信息
        order = indices[q_idx]
        g_pids = image_pids[order]
        g_camids = image_camids[order]
        
        # 匹配的位置
        matches = (g_pids == q_pid).astype(np.int32)
        
        # 计算 CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max(topk)])
        
        # 计算 AP
        num_rel = matches.sum()
        if num_rel > 0:
            tmp_cmc = matches.cumsum()
            tmp_cmc = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
            tmp_cmc = tmp_cmc * matches
            ap = tmp_cmc.sum() / num_rel
            all_ap.append(ap)
    
    all_cmc = np.array(all_cmc)
    mAP = np.mean(all_ap)
    
    # 计算平均 CMC
    cmc = all_cmc.mean(axis=0)
    
    metrics = {'mAP': mAP * 100}
    for k in topk:
        if k <= len(cmc):
            metrics[f'Rank-{k}'] = cmc[k-1] * 100
    
    return metrics


def evaluate_text_reid(
    model,
    query_loader,
    gallery_loader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    完整的 Text-to-Image ReID 评估
    
    Args:
        model: 模型（需要有 encode_text 和 encode_image 方法）
        query_loader: 查询集 DataLoader（使用文本查询）
        gallery_loader: 图库 DataLoader
        device: 设备
    
    Returns:
        metrics: 评估指标
    """
    print("=" * 50)
    print("Text-Guided ReID Evaluation")
    print("=" * 50)
    
    # 提取文本查询特征
    text_features, text_pids, texts = extract_text_features(
        model, query_loader, device
    )
    print(f"Query: {len(text_pids)} text queries")
    
    # 提取图库图像特征
    image_features, image_pids, image_camids = extract_image_features(
        model, gallery_loader, device
    )
    print(f"Gallery: {len(image_pids)} images")
    
    # 计算指标
    metrics = compute_text_to_image_metrics(
        text_features, text_pids,
        image_features, image_pids, image_camids,
        topk=[1, 5, 10]
    )
    
    print("\n=== Results ===")
    print(f"mAP: {metrics['mAP']:.2f}%")
    print(f"Rank-1: {metrics['Rank-1']:.2f}%")
    print(f"Rank-5: {metrics['Rank-5']:.2f}%")
    print(f"Rank-10: {metrics['Rank-10']:.2f}%")
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Text-Guided ReID')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--query_annotation', type=str, required=True)
    parser.add_argument('--gallery_annotation', type=str, required=True)
    
    args = parser.parse_args()
    
    # TODO: 实现完整的评估流程
    print("请实现完整的评估流程")
```

**Step 2: Commit**

```bash
git add scripts/evaluate_text_reid.py
git commit -m "feat: add text-to-image reid evaluation script"
```

---

## Summary

完成所有任务后，运行最终 commit:

```bash
git push origin exp/text-guided-reid
```

**总结**:
- Task 1: 下载属性标注
- Task 2: 属性解析器（方案 A）
- Task 3: BLIP 生成器（方案 B）
- Task 4: 文本数据集加载器
- Task 5: TextQueryEncoder 模块
- Task 6: 跨模态损失函数
- Task 7: 评估脚本

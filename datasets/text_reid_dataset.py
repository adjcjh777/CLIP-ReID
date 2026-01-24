# datasets/text_reid_dataset.py
"""
Text-Guided ReID 数据集加载器

支持:
1. Text-to-Image: 用文本查询图像
2. Image-to-Text: 用图像查询文本（对比学习训练用）
3. 标准的 Image-to-Image ReID（兼容原有模式）

Usage:
    from datasets.text_reid_dataset import TextReIDDataset, build_text_reid_dataloaders
    
    train_loader, query_loader, gallery_loader = build_text_reid_dataloaders(
        train_image_dir='/path/to/train',
        train_annotation='annotations/train.json',
        ...
    )
"""

import json
import random
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TextReIDDataset(Dataset):
    """
    Text-Guided ReID 数据集
    
    支持两种模式:
    1. text-to-image: 用文本查询图像
    2. image-to-text: 用图像查询文本
    
    数据格式 (JSON):
    {
        "image_name.jpg": {
            "pid": 123,
            "camid": 1,
            "text": "A young man wearing...",
            "source": "attribute" or "blip2"
        },
        ...
    }
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform=None,
        mode: str = 'train',  # 'train', 'query', 'gallery'
        return_text: bool = True,
    ):
        """
        Args:
            image_dir: 图像目录路径
            annotation_file: 文本标注 JSON 文件路径
            transform: 图像变换
            mode: 数据集模式
            return_text: 是否返回文本（gallery 模式可能不需要）
        """
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.return_text = return_text
        
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
                    'text': ann.get('text', ''),
                    'camid': ann.get('camid', self._parse_camid(img_name)),
                    'source': ann.get('source', 'unknown')
                })
        
        # 按 PID 排序，保证相同身份的样本在一起
        self.data.sort(key=lambda x: (x['pid'], x['image_name']))
        
        # 构建 PID 到索引的映射
        self.pid_to_indices = {}
        for idx, item in enumerate(self.data):
            pid = item['pid']
            if pid not in self.pid_to_indices:
                self.pid_to_indices[pid] = []
            self.pid_to_indices[pid].append(idx)
        
        # 重新映射 PID 到连续的数字（用于分类器）
        unique_pids = sorted(self.pid_to_indices.keys())
        self.pid_to_label = {pid: label for label, pid in enumerate(unique_pids)}
        self.label_to_pid = {label: pid for pid, label in self.pid_to_label.items()}
        self.num_classes = len(unique_pids)
        
        # 为每个样本添加 label
        for item in self.data:
            item['label'] = self.pid_to_label[item['pid']]
        
        # 图像变换
        if transform is None:
            if mode == 'train':
                self.transform = T.Compose([
                    T.Resize((256, 128)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Pad(10),
                    T.RandomCrop((256, 128)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((256, 128)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform
        
        print(f"TextReIDDataset [{mode}]: {len(self.data)} samples, "
              f"{self.num_classes} identities")
    
    def _parse_camid(self, img_name: str) -> int:
        """解析相机 ID，如 0001_c1s1_000151_01.jpg -> 1"""
        try:
            return int(img_name.split('_')[1][1])
        except:
            return 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        result = {
            'image': image,
            'pid': item['pid'],
            'label': item['label'],
            'camid': item['camid'],
            'image_path': item['image_path'],
            'image_name': item['image_name']
        }
        
        if self.return_text:
            result['text'] = item['text']
        
        return result
    
    def get_text_by_pid(self, pid: int) -> List[str]:
        """获取指定身份的所有文本描述"""
        texts = []
        for idx in self.pid_to_indices.get(pid, []):
            texts.append(self.data[idx]['text'])
        return texts
    
    def get_unique_text_per_pid(self) -> Dict[int, str]:
        """为每个身份获取一个代表性文本（用于 text query）"""
        text_per_pid = {}
        for pid, indices in self.pid_to_indices.items():
            # 选择第一个非空文本
            for idx in indices:
                text = self.data[idx]['text']
                if text:
                    text_per_pid[pid] = text
                    break
        return text_per_pid
    
    def get_all_pids(self) -> List[int]:
        """获取所有身份 ID"""
        return list(self.pid_to_indices.keys())


class TextReIDCollator:
    """
    自定义 collate_fn，处理文本和图像的批处理
    """
    
    def __init__(self, tokenizer=None, max_length: int = 77):
        """
        Args:
            tokenizer: CLIP tokenizer 或其他 tokenizer
            max_length: 文本最大长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        pids = torch.tensor([item['pid'] for item in batch], dtype=torch.long)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        camids = torch.tensor([item['camid'] for item in batch], dtype=torch.long)
        image_paths = [item['image_path'] for item in batch]
        
        result = {
            'images': images,
            'pids': pids,
            'labels': labels,
            'camids': camids,
            'image_paths': image_paths
        }
        
        # 处理文本
        if 'text' in batch[0]:
            texts = [item['text'] for item in batch]
            result['texts'] = texts
            
            # 如果有 tokenizer，则进行文本编码
            if self.tokenizer is not None:
                text_tokens = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                result['text_tokens'] = text_tokens
        
        return result


class IdentitySampler:
    """
    身份采样器：确保每个 batch 包含 P 个身份，每个身份 K 个样本
    用于 triplet loss 训练
    """
    
    def __init__(
        self,
        dataset: TextReIDDataset,
        batch_size: int,
        num_instances: int = 4
    ):
        """
        Args:
            dataset: TextReIDDataset 实例
            batch_size: 批次大小 (应该是 P * K)
            num_instances: 每个身份的样本数 K
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        self.pids = list(dataset.pid_to_indices.keys())
        self.pid_to_indices = dataset.pid_to_indices
        
    def __iter__(self):
        indices = []
        
        # 打乱身份顺序
        random.shuffle(self.pids)
        
        for pid in self.pids:
            pid_indices = self.pid_to_indices[pid]
            
            if len(pid_indices) >= self.num_instances:
                # 随机选择 K 个样本
                selected = random.sample(pid_indices, self.num_instances)
            else:
                # 如果样本不够，重复采样
                selected = random.choices(pid_indices, k=self.num_instances)
            
            indices.extend(selected)
            
            # 当收集够一个 batch 时，yield
            if len(indices) >= self.batch_size:
                batch_indices = indices[:self.batch_size]
                indices = indices[self.batch_size:]
                yield batch_indices
        
        # 处理剩余的样本
        if indices:
            yield indices
    
    def __len__(self):
        return len(self.pids) * self.num_instances // self.batch_size


def build_text_reid_dataloaders(
    train_image_dir: str,
    train_annotation: str,
    query_image_dir: str,
    query_annotation: str,
    gallery_image_dir: str,
    gallery_annotation: str,
    batch_size: int = 64,
    num_instances: int = 4,
    num_workers: int = 4,
    tokenizer=None,
    use_identity_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建 Text ReID 数据加载器
    
    Args:
        train_image_dir: 训练图像目录
        train_annotation: 训练集标注文件
        query_image_dir: 查询图像目录
        query_annotation: 查询集标注文件
        gallery_image_dir: 图库图像目录
        gallery_annotation: 图库标注文件
        batch_size: 批次大小
        num_instances: 每个身份的样本数（用于 triplet）
        num_workers: 数据加载线程数
        tokenizer: 文本 tokenizer
        use_identity_sampler: 是否使用身份采样器
    
    Returns:
        train_loader, query_loader, gallery_loader
    """
    # 创建数据集
    train_dataset = TextReIDDataset(
        train_image_dir, train_annotation, mode='train'
    )
    query_dataset = TextReIDDataset(
        query_image_dir, query_annotation, mode='query'
    )
    gallery_dataset = TextReIDDataset(
        gallery_image_dir, gallery_annotation, mode='gallery', return_text=False
    )
    
    collator = TextReIDCollator(tokenizer)
    
    # 训练集使用身份采样器
    if use_identity_sampler:
        sampler = IdentitySampler(train_dataset, batch_size, num_instances)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    else:
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


def quick_test():
    """快速测试数据集功能"""
    import tempfile
    import os
    
    # 创建临时测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建假图像
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        
        for i in range(10):
            pid = i // 2 + 1
            img = Image.new('RGB', (128, 256), color=(i * 25, i * 20, i * 15))
            img.save(img_dir / f"{pid:04d}_c1s1_{i:06d}_01.jpg")
        
        # 创建假标注
        annotations = {}
        for img_path in img_dir.glob("*.jpg"):
            pid = int(img_path.name.split('_')[0])
            annotations[img_path.name] = {
                'pid': pid,
                'camid': 1,
                'text': f"A person with ID {pid} wearing casual clothes.",
                'source': 'test'
            }
        
        ann_file = Path(tmpdir) / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(annotations, f)
        
        # 测试数据集
        dataset = TextReIDDataset(str(img_dir), str(ann_file), mode='train')
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Num classes: {dataset.num_classes}")
        
        # 测试 __getitem__
        item = dataset[0]
        print(f"Image shape: {item['image'].shape}")
        print(f"Text: {item['text']}")
        print(f"PID: {item['pid']}, Label: {item['label']}")
        
        print("\nQuick test passed!")


if __name__ == '__main__':
    quick_test()

# datasets/text_reid_dataset.py
import json
import random
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class TextReIDDataset(Dataset):
    """
    Text-Guided ReID 数据集
    支持 image-to-text 和 text-to-image 训练
    支持 Market1501 (平铺) 和 MSMT17 (子目录) 两种结构
    """
    def __init__(self, image_dir: str, annotation_file: str, transform=None, mode: str = 'train'):
        self.image_dir = Path(image_dir)
        self.mode = mode
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.data = []
        for img_name, ann in self.annotations.items():
            # 尝试直接路径 (Market1501 风格)
            img_path = self.image_dir / img_name
            
            # 如果不存在，尝试子目录结构 (MSMT17 风格: pid/img_name)
            if not img_path.exists():
                pid_str = str(ann['pid']).zfill(4)
                img_path = self.image_dir / pid_str / img_name
            
            if img_path.exists():
                self.data.append({
                    'image_path': str(img_path),
                    'pid': ann['pid'],
                    'text': ann['text'],
                    'camid': ann.get('camid', 0)
                })
        
        print(f"Loaded {len(self.data)} images from {image_dir}")
        
        # PID 重映射 (0 ~ N-1)
        unique_pids = sorted(list(set(d['pid'] for d in self.data)))
        self.pid_to_label = {pid: i for i, pid in enumerate(unique_pids)}
        self.num_classes = len(unique_pids)
        
        # 默认变换
        if transform is None:
            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        label = self.pid_to_label[item['pid']]
        
        return {
            'image': image,
            'text': item['text'],
            'pid': item['pid'],
            'label': label,
            'camid': item['camid']
        }

class TextReIDCollator:
    """处理文本 tokenization"""
    def __init__(self, tokenizer=None, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        images = torch.stack([x['image'] for x in batch])
        labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
        pids = torch.tensor([x['pid'] for x in batch], dtype=torch.long)
        camids = torch.tensor([x['camid'] for x in batch], dtype=torch.long)
        texts = [x['text'] for x in batch]
        
        res = {
            'images': images,
            'labels': labels,
            'pids': pids,
            'camids': camids,
            'texts': texts
        }
        
        if self.tokenizer:
            text_tokens = self.tokenizer(texts).squeeze(1) # [B, 77]
            res['text_tokens'] = text_tokens
            
        return res

def make_text_dataloader(cfg, annotation_file, tokenizer=None):
    from .sampler import RandomIdentitySampler
    
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    try:
        from timm.data.random_erasing import RandomErasing
        train_transforms.transforms.append(
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
        )
    except ImportError:
        pass

    # 根据数据集名称选择图像目录
    dataset_name = cfg.DATASETS.NAMES
    if 'msmt17' in dataset_name.lower():
        image_dir = cfg.DATASETS.ROOT_DIR + '/MSMT17/train'
    else:  # market1501 或其他
        image_dir = cfg.DATASETS.ROOT_DIR + '/Market-1501-v15.09.15/bounding_box_train'
    
    dataset = TextReIDDataset(
        image_dir,
        annotation_file,
        transform=train_transforms
    )
    # RandomIdentitySampler 需要 data_source 是一个列表，每个元素是 4 元组
    data_source = [(d['image_path'], d['pid'], d['camid'], 0) for d in dataset.data]  # viewid 设为 0
    
    collator = TextReIDCollator(tokenizer)
    
    # 使用 RandomIdentitySampler 保证每个 batch 有 NUM_INSTANCE 个相同 ID 的样本
    loader = DataLoader(
        dataset,
        batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(data_source, cfg.SOLVER.STAGE1.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True
    )
    
    return loader, dataset.num_classes

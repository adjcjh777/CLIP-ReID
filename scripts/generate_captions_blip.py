# scripts/generate_captions_blip.py
"""
使用 BLIP-2 为 ReID 数据集生成文本描述

方案 B: 使用视觉语言模型自动生成描述

Usage:
    python scripts/generate_captions_blip.py \
        --image_dir /path/to/images \
        --output_file annotations.json \
        --batch_size 16

Requirements:
    pip install transformers accelerate

Notes:
    - 默认使用 BLIP2 模型（更强但更重）。
    - 如果显存不足，建议使用 BLIP1：
      Salesforce/blip-image-captioning-base
      Salesforce/blip-image-captioning-large
"""

import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional


class ImageDataset(Dataset):
    """简单的图像数据集"""
    
    def __init__(self, image_dir: str, processor=None):
        self.image_dir = Path(image_dir)
        self.processor = processor
        
        # 收集所有图像
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        
        # 过滤掉无效图像（如 -1 开头的 junk 图像）
        self.image_paths = [
            p for p in self.image_paths 
            if not p.name.startswith('-1') and not p.name.startswith('0000')
        ]
        
        print(f"Found {len(self.image_paths)} valid images in {image_dir}")
    
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
        
        # 解析 PID 和 CamID
        img_name = img_path.name
        try:
            pid = int(img_name.split('_')[0])
            camid = int(img_name.split('_')[1][1])
        except:
            pid = -1
            camid = 0
        
        return {
            'image': image,
            'path': str(img_path),
            'name': img_name,
            'pid': pid,
            'camid': camid
        }


def collate_fn(batch):
    """自定义 collate_fn"""
    images = [item['image'] for item in batch]
    paths = [item['path'] for item in batch]
    names = [item['name'] for item in batch]
    pids = [item['pid'] for item in batch]
    camids = [item['camid'] for item in batch]
    return images, paths, names, pids, camids


def _load_model(model_name: str, device: str):
    """Load BLIP/BLIP2 model based on model_name."""
    use_blip2 = "blip2" in model_name.lower()
    if use_blip2:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        device_map = "auto" if device.startswith("cuda") else None
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map
        )
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        if device.startswith("cuda"):
            model = model.half()
    if not (use_blip2 and device.startswith("cuda")):
        model = model.to(device)
    model.eval()
    return processor, model, use_blip2


def generate_captions(
    image_dir: str,
    output_file: str,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    batch_size: int = 8,
    device: str = "cuda",
    prompt: str = None,
    max_images: Optional[int] = None,
):
    """
    使用 BLIP-2 生成图像描述
    
    Args:
        image_dir: 图像目录
        output_file: 输出 JSON 文件
        model_name: BLIP-2 模型名称
        batch_size: 批次大小
        device: 设备
        prompt: 自定义提示词
    """
    print(f"=== BLIP Caption Generation ===")
    print(f"Model: {model_name}")
    print(f"Image dir: {image_dir}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size}")
    print()
    
    # 加载模型
    print("Loading BLIP model...")
    processor, model, use_blip2 = _load_model(model_name, device)
    print("Model loaded successfully!")
    
    # 创建数据集
    dataset = ImageDataset(image_dir, processor)
    if max_images is not None:
        dataset.image_paths = dataset.image_paths[:max_images]
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 用于 ReID 任务的提示词
    if prompt is None:
        prompt = (
            "Describe this person's appearance in detail, including their "
            "gender, age, clothing colors, clothing style, and any accessories "
            "they are carrying."
        )
    
    print(f"Using prompt: {prompt}")
    print()
    
    annotations = {}
    
    print("Generating captions...")
    for images, paths, names, pids, camids in tqdm(dataloader):
        # 处理输入
        inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device.startswith("cuda") and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        
        # 生成描述
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=80,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # 解码
        generated_texts = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        # 保存结果
        for name, path, text, pid, camid in zip(
            names, paths, generated_texts, pids, camids
        ):
            # 清理生成的文本
            text = text.strip()
            # 移除可能的重复提示词
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            
            annotations[name] = {
                'pid': pid,
                'camid': camid,
                'text': text,
                'source': 'blip2' if use_blip2 else 'blip',
                'model': model_name
            }
    
    # 保存到文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Summary ===")
    print(f"Generated {len(annotations)} captions")
    print(f"Saved to {output_file}")
    
    # 打印一些示例
    print(f"\n=== Sample Captions ===")
    for i, (name, ann) in enumerate(list(annotations.items())[:5]):
        print(f"{name}:")
        print(f"  PID: {ann['pid']}")
        print(f"  Text: {ann['text'][:100]}...")
        print()


def generate_captions_batch(
    image_dirs: List[str],
    output_dir: str,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    batch_size: int = 8,
    device: str = "cuda"
):
    """
    批量处理多个目录
    
    Args:
        image_dirs: 图像目录列表
        output_dir: 输出目录
        model_name: 模型名称
        batch_size: 批次大小
        device: 设备
    """
    # 只加载一次模型
    print("Loading BLIP model...")
    processor, model, use_blip2 = _load_model(model_name, device)
    print("Model loaded!\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for image_dir in image_dirs:
        dir_name = Path(image_dir).name
        output_file = output_path / f"{dir_name}_captions.json"
        
        print(f"Processing {image_dir}...")
        
        dataset = ImageDataset(image_dir, processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        prompt = "Describe this person's appearance including clothing and accessories."
        
        annotations = {}
        
        for images, paths, names, pids, camids in tqdm(dataloader, desc=dir_name):
            inputs = processor(
                images=images,
                text=[prompt] * len(images),
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device.startswith("cuda") and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    early_stopping=True
                )
            
            generated_texts = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            for name, text, pid, camid in zip(names, generated_texts, pids, camids):
                annotations[name] = {
                    'pid': pid,
                    'camid': camid,
                    'text': text.strip(),
                    'source': 'blip2' if use_blip2 else 'blip',
                    'model': model_name
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(annotations)} annotations to {output_file}\n")


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
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for generation')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit number of images for quick test')
    
    args = parser.parse_args()
    
    generate_captions(
        image_dir=args.image_dir,
        output_file=args.output_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        prompt=args.prompt,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()

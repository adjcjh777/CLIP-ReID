#!/usr/bin/env python
"""为 MSMT17 数据集生成文本标注"""
import os
import json
import argparse
from pathlib import Path

def parse_msmt17_filename(filename):
    """解析 MSMT17 文件名
    格式: {pid}_{序号}_{camid}_{时间}_{...}.jpg
    """
    parts = filename.split('_')
    pid = int(parts[0])
    camid = int(parts[2])
    return pid, camid

def generate_msmt17_annotations(train_dir, output_file):
    """为 MSMT17 训练集生成标注"""
    annotations = {}
    
    train_path = Path(train_dir)
    
    # 遍历所有子目录 (每个目录是一个 PID)
    for pid_dir in sorted(train_path.iterdir()):
        if not pid_dir.is_dir():
            continue
            
        pid = int(pid_dir.name)
        
        for img_file in pid_dir.iterdir():
            if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                continue
                
            try:
                _, camid = parse_msmt17_filename(img_file.name)
            except:
                continue
            
            # 生成简单的文本描述
            # 由于 MSMT17 没有属性标注，我们使用通用描述
            text = f"A person captured by camera {camid}. Person ID {pid}."
            
            annotations[img_file.name] = {
                'pid': pid,
                'camid': camid,
                'text': text
            }
    
    # 保存标注
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Generated {len(annotations)} annotations for MSMT17")
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/root/autodl-tmp/CLIP_REID/DATASETS/MSMT17/train')
    parser.add_argument('--output_file', default='annotations/msmt17_train.json')
    args = parser.parse_args()
    
    generate_msmt17_annotations(args.train_dir, args.output_file)

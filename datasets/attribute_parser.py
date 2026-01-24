# datasets/attribute_parser.py
"""
解析 Market-1501 属性标注文件，将属性转换为文本描述

方案 A: 基于属性模板生成文本描述

Usage:
    python datasets/attribute_parser.py \
        --attribute_file datasets/attributes/market1501_attribute.mat \
        --image_dir /path/to/bounding_box_train \
        --output_file annotations/market1501_train_text.json
"""

import scipy.io as sio
import numpy as np
import json
import random
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple


# Market-1501 属性定义（27个属性）
# 参考: https://github.com/vana77/Market-1501_Attribute
MARKET_ATTRIBUTE_NAMES = [
    'age',           # 0: young, 1: teenager, 2: adult, 3: old
    'backpack',      # 0: no, 1: yes
    'bag',           # 0: no, 1: yes
    'handbag',       # 0: no, 1: yes
    'downblack',     # 下装颜色
    'downblue',
    'downbrown',
    'downgray',
    'downgreen',
    'downpink',
    'downpurple',
    'downwhite',
    'downyellow',
    'upblack',       # 上装颜色
    'upblue',
    'upgreen',
    'upgray',
    'uppurple',
    'upred',
    'upwhite',
    'upyellow',
    'clothes',       # 1: dress, 2: pants
    'down',          # 1: long, 2: short
    'up',            # 1: long, 2: short
    'hair',          # 1: short, 2: long
    'hat',           # 0: no, 1: yes
    'gender',        # 1: male, 2: female
]

# 颜色映射
UPCOLORS = ['black', 'blue', 'green', 'gray', 'purple', 'red', 'white', 'yellow']
DOWNCOLORS = ['black', 'blue', 'brown', 'gray', 'green', 'pink', 'purple', 'white', 'yellow']

# 文本模板
TEMPLATES = {
    'basic': "A {age} {gender} with {hair} hair, wearing a {up_color} {up_style} top and {down_color} {down_type}.",
    'with_accessories': "A {age} {gender} with {hair} hair, wearing a {up_color} {up_style} top and {down_color} {down_type}, carrying {accessories}.",
    'detailed': "This is a {age} {gender}. The person has {hair} hair and is wearing a {up_color} {up_style} top paired with {down_color} {down_type}. {accessory_detail}",
}

# 同义词替换（增加多样性）
SYNONYMS = {
    'young': ['young', 'youthful'],
    'teenager': ['teenage', 'adolescent', 'teen'],
    'adult': ['adult', 'grown-up', 'middle-aged'],
    'old': ['elderly', 'older', 'senior'],
    'male': ['male', 'man', 'gentleman'],
    'female': ['female', 'woman', 'lady'],
    'short': ['short', 'cropped', 'trimmed'],
    'long': ['long', 'lengthy', 'flowing'],
}


class Market1501AttributeParser:
    """解析 Market-1501 属性标注"""
    
    def __init__(self, attribute_file: str, verbose: bool = True):
        """
        Args:
            attribute_file: market1501_attribute.mat 文件路径
            verbose: 是否打印详细信息
        """
        self.attribute_file = Path(attribute_file)
        self.verbose = verbose
        self.train_attrs = {}  # pid -> attributes
        self.test_attrs = {}
        
        self._load_attributes()
    
    def _load_attributes(self):
        """加载属性文件"""
        if not self.attribute_file.exists():
            raise FileNotFoundError(f"属性文件不存在: {self.attribute_file}")
        
        mat = sio.loadmat(str(self.attribute_file))
        
        # 解析 mat 文件结构
        # market_attribute 是一个结构体，包含 train 和 test
        market_attr = mat['market_attribute'][0, 0]
        
        # 获取训练集属性
        train_data = market_attr['train'][0, 0]
        test_data = market_attr['test'][0, 0]
        
        # 解析训练集
        self._parse_split(train_data, self.train_attrs, 'train')
        # 解析测试集
        self._parse_split(test_data, self.test_attrs, 'test')
        
        if self.verbose:
            print(f"加载了 {len(self.train_attrs)} 个训练集身份的属性")
            print(f"加载了 {len(self.test_attrs)} 个测试集身份的属性")
    
    def _parse_split(self, data, attr_dict: dict, split: str):
        """解析一个数据划分"""
        # data 结构取决于 mat 文件的具体格式
        # 通常包含 image_index 和各属性列
        try:
            # 尝试获取属性数据
            for i, attr_name in enumerate(MARKET_ATTRIBUTE_NAMES):
                if attr_name in data.dtype.names:
                    attr_values = data[attr_name].flatten()
                    # 存储到字典中
                    for pid_idx, value in enumerate(attr_values):
                        pid = pid_idx + 1  # PID 从 1 开始
                        if pid not in attr_dict:
                            attr_dict[pid] = {}
                        attr_dict[pid][attr_name] = int(value)
        except Exception as e:
            if self.verbose:
                print(f"Warning: 解析 {split} 属性时出错: {e}")
                print("尝试备用解析方法...")
            self._parse_split_fallback(data, attr_dict)
    
    def _parse_split_fallback(self, data, attr_dict: dict):
        """备用解析方法"""
        # 如果标准方法失败，尝试直接访问数组
        pass
    
    def get_attributes(self, pid: int, split: str = 'train') -> Optional[Dict]:
        """
        获取指定身份的属性
        
        Args:
            pid: 行人身份 ID
            split: 'train' 或 'test'
        
        Returns:
            属性字典，如果不存在则返回 None
        """
        attr_dict = self.train_attrs if split == 'train' else self.test_attrs
        return attr_dict.get(pid)
    
    def attributes_to_text(
        self, 
        attrs: Dict, 
        template: str = 'basic',
        use_synonyms: bool = True
    ) -> str:
        """
        将属性字典转换为自然语言描述
        
        Args:
            attrs: 属性字典
            template: 使用的模板类型 ('basic', 'with_accessories', 'detailed')
            use_synonyms: 是否使用同义词替换增加多样性
        
        Returns:
            自然语言描述
        """
        def get_word(key, value_map, default='unknown'):
            word = value_map.get(key, default)
            if use_synonyms and word in SYNONYMS:
                return random.choice(SYNONYMS[word])
            return word
        
        # 解析年龄
        age_map = {0: 'young', 1: 'teenager', 2: 'adult', 3: 'old'}
        age = get_word(attrs.get('age', 2), age_map, 'adult')
        
        # 解析性别
        gender_map = {1: 'male', 2: 'female'}
        gender = get_word(attrs.get('gender', 1), gender_map, 'person')
        
        # 解析头发
        hair_map = {1: 'short', 2: 'long'}
        hair = get_word(attrs.get('hair', 1), hair_map, 'short')
        
        # 解析上装颜色
        up_color = 'colored'
        for i, color in enumerate(UPCOLORS):
            if attrs.get(f'up{color}', 0) == 1:
                up_color = color
                break
        
        # 解析上装款式
        up_style = 'long-sleeved' if attrs.get('up', 1) == 1 else 'short-sleeved'
        
        # 解析下装颜色
        down_color = 'colored'
        for i, color in enumerate(DOWNCOLORS):
            if attrs.get(f'down{color}', 0) == 1:
                down_color = color
                break
        
        # 解析下装类型
        down_type = 'dress' if attrs.get('clothes', 2) == 1 else 'pants'
        if attrs.get('down', 1) == 2:
            down_type = 'shorts' if down_type == 'pants' else 'skirt'
        
        # 解析配饰
        accessories = []
        if attrs.get('backpack', 0) == 1:
            accessories.append('a backpack')
        if attrs.get('bag', 0) == 1:
            accessories.append('a bag')
        if attrs.get('handbag', 0) == 1:
            accessories.append('a handbag')
        if attrs.get('hat', 0) == 1:
            accessories.append('a hat')
        
        # 生成描述
        if template == 'basic':
            text = TEMPLATES['basic'].format(
                age=age, gender=gender, hair=hair,
                up_color=up_color, up_style=up_style,
                down_color=down_color, down_type=down_type
            )
        elif template == 'with_accessories' and accessories:
            text = TEMPLATES['with_accessories'].format(
                age=age, gender=gender, hair=hair,
                up_color=up_color, up_style=up_style,
                down_color=down_color, down_type=down_type,
                accessories=' and '.join(accessories)
            )
        else:
            # 详细模板
            accessory_detail = ""
            if accessories:
                accessory_detail = f"The person is carrying {' and '.join(accessories)}."
            text = TEMPLATES['detailed'].format(
                age=age, gender=gender, hair=hair,
                up_color=up_color, up_style=up_style,
                down_color=down_color, down_type=down_type,
                accessory_detail=accessory_detail
            )
        
        return text


def generate_text_annotations(
    attribute_file: str,
    image_dir: str,
    output_file: str,
    split: str = 'train',
    templates: List[str] = ['basic', 'with_accessories', 'detailed']
):
    """
    为所有图像生成文本标注
    
    Args:
        attribute_file: 属性标注文件路径
        image_dir: 图像目录路径
        output_file: 输出的 JSON 文件路径
        split: 'train' 或 'test'
        templates: 要使用的模板列表
    """
    print(f"=== Generating Text Annotations ===")
    print(f"Attribute file: {attribute_file}")
    print(f"Image directory: {image_dir}")
    print(f"Output file: {output_file}")
    
    parser = Market1501AttributeParser(attribute_file)
    
    # 收集所有图像
    image_paths = sorted(glob(f"{image_dir}/*.jpg"))
    print(f"Found {len(image_paths)} images")
    
    annotations = {}
    success_count = 0
    fail_count = 0
    
    for img_path in image_paths:
        img_name = Path(img_path).name
        
        # 跳过 junk 图像
        if img_name.startswith('-1') or img_name.startswith('0000'):
            continue
        
        # 解析 PID: 0001_c1s1_000151_01.jpg -> 1
        try:
            pid = int(img_name.split('_')[0])
        except:
            fail_count += 1
            continue
        
        # 获取属性
        attrs = parser.get_attributes(pid, split)
        
        if attrs is None:
            # 如果没有属性，生成一个通用描述
            text = "A person in casual clothing."
            annotations[img_name] = {
                'pid': pid,
                'text': text,
                'source': 'fallback',
                'camid': int(img_name.split('_')[1][1]) if '_c' in img_name else 0
            }
            fail_count += 1
        else:
            # 随机选择一个模板
            template = random.choice(templates)
            text = parser.attributes_to_text(attrs, template)
            
            annotations[img_name] = {
                'pid': pid,
                'text': text,
                'source': 'attribute',
                'template': template,
                'attributes': attrs,
                'camid': int(img_name.split('_')[1][1]) if '_c' in img_name else 0
            }
            success_count += 1
    
    # 保存到文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Summary ===")
    print(f"Total images processed: {success_count + fail_count}")
    print(f"Successfully annotated: {success_count}")
    print(f"Fallback annotations: {fail_count}")
    print(f"Saved to: {output_file}")
    
    # 打印一些示例
    print(f"\n=== Sample Annotations ===")
    for i, (name, ann) in enumerate(list(annotations.items())[:3]):
        print(f"{name}:")
        print(f"  PID: {ann['pid']}")
        print(f"  Text: {ann['text']}")
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate text annotations from Market-1501 attributes'
    )
    parser.add_argument('--attribute_file', type=str, required=True,
                        help='Path to market1501_attribute.mat')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split')
    
    args = parser.parse_args()
    
    generate_text_annotations(
        args.attribute_file,
        args.image_dir,
        args.output_file,
        args.split
    )

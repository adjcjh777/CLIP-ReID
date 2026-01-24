# scripts/evaluate_text_reid.py
"""
Text-Guided ReID 评估脚本

支持:
1. Text-to-Image Retrieval: 用文本查询图像
2. 计算 Rank-1, Rank-5, Rank-10, mAP 指标
3. 可视化检索结果

Usage:
    python scripts/evaluate_text_reid.py \
        --config_file configs/text_reid.yml \
        --model_weights output/model.pth \
        --query_annotation annotations/query.json \
        --gallery_annotation annotations/gallery.json
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_image_features(
    model,
    dataloader,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[str]]:
    """
    提取所有图像特征
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        features: [N, D] 图像特征
        pids: [N] 身份 ID
        camids: [N] 相机 ID
        paths: [N] 图像路径
    """
    model.eval()
    
    all_features = []
    all_pids = []
    all_camids = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting image features'):
            images = batch['images'].to(device)
            
            # 提取特征
            if hasattr(model, 'encode_image'):
                features = model.encode_image(images)
            else:
                # 兼容原始 CLIP-ReID 模型
                features = model(x=images)
            
            features = F.normalize(features, p=2, dim=-1)
            
            all_features.append(features.cpu())
            all_pids.extend(batch['pids'].numpy())
            all_camids.extend(batch['camids'].numpy())
            all_paths.extend(batch['image_paths'])
    
    all_features = torch.cat(all_features, dim=0)
    all_pids = np.array(all_pids)
    all_camids = np.array(all_camids)
    
    return all_features, all_pids, all_camids, all_paths


def extract_text_features(
    model,
    dataloader,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """
    提取所有文本特征
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
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
            # 获取文本 tokens
            text_tokens = batch.get('text_tokens')
            
            if text_tokens is None:
                # 如果没有预先 tokenize，需要在这里处理
                texts = batch['texts']
                # 这里需要 tokenizer，暂时跳过
                continue
            
            if isinstance(text_tokens, dict):
                input_ids = text_tokens['input_ids'].to(device)
            else:
                input_ids = text_tokens.to(device)
            
            # 提取特征
            if hasattr(model, 'encode_text'):
                features = model.encode_text(input_ids)
            elif hasattr(model, 'text_encoder'):
                features = model.text_encoder(input_ids)
            else:
                raise ValueError("Model does not have text encoding capability")
            
            features = F.normalize(features, p=2, dim=-1)
            
            all_features.append(features.cpu())
            all_pids.extend(batch['pids'].numpy())
            all_texts.extend(batch['texts'])
    
    all_features = torch.cat(all_features, dim=0)
    all_pids = np.array(all_pids)
    
    return all_features, all_pids, all_texts


def compute_metrics(
    query_features: torch.Tensor,
    query_pids: np.ndarray,
    gallery_features: torch.Tensor,
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    topk: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算检索指标
    
    Args:
        query_features: 查询特征 [Q, D]
        query_pids: 查询身份 [Q]
        gallery_features: 图库特征 [G, D]
        gallery_pids: 图库身份 [G]
        gallery_camids: 图库相机 ID [G]
        topk: 计算 Rank-k 的 k 值列表
    
    Returns:
        metrics: 包含 Rank-k 和 mAP 的字典
    """
    num_query = query_features.shape[0]
    
    # 计算相似度矩阵
    similarity = query_features @ gallery_features.T  # [Q, G]
    similarity = similarity.numpy()
    
    # 排序（降序）
    indices = np.argsort(-similarity, axis=1)
    
    # 计算指标
    all_cmc = []
    all_ap = []
    
    for q_idx in range(num_query):
        q_pid = query_pids[q_idx]
        
        # 获取排序后的 gallery 信息
        order = indices[q_idx]
        g_pids = gallery_pids[order]
        
        # 匹配标记
        matches = (g_pids == q_pid).astype(np.int32)
        
        # 如果没有正样本，跳过
        if matches.sum() == 0:
            continue
        
        # 计算 CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max(topk)])
        
        # 计算 AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        recall_change = matches
        ap = (precision * recall_change).sum() / num_rel
        all_ap.append(ap)
    
    all_cmc = np.array(all_cmc)
    mAP = np.mean(all_ap) if all_ap else 0.0
    
    # 计算平均 CMC
    cmc = all_cmc.mean(axis=0) if len(all_cmc) > 0 else np.zeros(max(topk))
    
    metrics = {'mAP': mAP * 100}
    for k in topk:
        if k <= len(cmc):
            metrics[f'Rank-{k}'] = cmc[k-1] * 100
    
    return metrics


def visualize_results(
    query_texts: List[str],
    query_pids: np.ndarray,
    gallery_paths: List[str],
    gallery_pids: np.ndarray,
    similarity: np.ndarray,
    save_dir: str,
    num_queries: int = 5,
    num_results: int = 10
):
    """
    可视化检索结果
    
    Args:
        query_texts: 查询文本列表
        query_pids: 查询身份
        gallery_paths: 图库图像路径
        gallery_pids: 图库身份
        similarity: 相似度矩阵 [Q, G]
        save_dir: 保存目录
        num_queries: 可视化的查询数量
        num_results: 每个查询显示的结果数量
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    indices = np.argsort(-similarity, axis=1)
    
    for q_idx in range(min(num_queries, len(query_texts))):
        q_text = query_texts[q_idx]
        q_pid = query_pids[q_idx]
        
        # 获取 top-k 结果
        topk_indices = indices[q_idx, :num_results]
        
        # 创建图形
        fig, axes = plt.subplots(1, num_results + 1, figsize=(20, 4))
        
        # 显示查询文本
        axes[0].text(0.5, 0.5, f"Query:\n{q_text[:50]}...\nPID: {q_pid}",
                     ha='center', va='center', wrap=True, fontsize=8)
        axes[0].axis('off')
        axes[0].set_title('Query Text', fontsize=10)
        
        # 显示检索结果
        for i, g_idx in enumerate(topk_indices):
            img_path = gallery_paths[g_idx]
            g_pid = gallery_pids[g_idx]
            sim = similarity[q_idx, g_idx]
            
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i + 1].imshow(img)
            except:
                axes[i + 1].text(0.5, 0.5, 'Image\nNot Found',
                               ha='center', va='center')
            
            # 设置标题（绿色=匹配，红色=不匹配）
            color = 'green' if g_pid == q_pid else 'red'
            axes[i + 1].set_title(f'PID:{g_pid}\nSim:{sim:.3f}', 
                                 fontsize=8, color=color)
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / f'query_{q_idx}.png', dpi=150)
        plt.close()
    
    print(f"Visualization saved to {save_path}")


def evaluate_text_to_image(
    model,
    query_loader,
    gallery_loader,
    device: str = 'cuda',
    visualize: bool = False,
    save_dir: str = None
) -> Dict[str, float]:
    """
    完整的 Text-to-Image ReID 评估
    
    Args:
        model: 模型
        query_loader: 查询集 DataLoader（使用文本查询）
        gallery_loader: 图库 DataLoader
        device: 设备
        visualize: 是否可视化结果
        save_dir: 可视化保存目录
    
    Returns:
        metrics: 评估指标
    """
    print("=" * 60)
    print("Text-Guided ReID Evaluation (Text-to-Image)")
    print("=" * 60)
    
    # 提取文本查询特征
    text_features, text_pids, texts = extract_text_features(
        model, query_loader, device
    )
    print(f"Query: {len(text_pids)} text queries")
    
    # 提取图库图像特征
    image_features, image_pids, image_camids, image_paths = extract_image_features(
        model, gallery_loader, device
    )
    print(f"Gallery: {len(image_pids)} images")
    
    # 计算指标
    metrics = compute_metrics(
        text_features, text_pids,
        image_features, image_pids, image_camids,
        topk=[1, 5, 10]
    )
    
    print("\n" + "=" * 40)
    print("Results:")
    print("=" * 40)
    print(f"mAP:     {metrics['mAP']:.2f}%")
    print(f"Rank-1:  {metrics['Rank-1']:.2f}%")
    print(f"Rank-5:  {metrics['Rank-5']:.2f}%")
    print(f"Rank-10: {metrics['Rank-10']:.2f}%")
    print("=" * 40)
    
    # 可视化
    if visualize and save_dir:
        similarity = text_features @ image_features.T
        similarity = similarity.numpy()
        
        visualize_results(
            texts, text_pids,
            image_paths, image_pids,
            similarity,
            save_dir
        )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Text-Guided ReID')
    parser.add_argument('--config_file', type=str, default='',
                        help='Path to config file')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--query_image_dir', type=str, required=True,
                        help='Query image directory')
    parser.add_argument('--query_annotation', type=str, required=True,
                        help='Query annotation file')
    parser.add_argument('--gallery_image_dir', type=str, required=True,
                        help='Gallery image directory')
    parser.add_argument('--gallery_annotation', type=str, required=True,
                        help='Gallery annotation file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save_dir', type=str, default='visualization',
                        help='Directory to save visualization')
    
    args = parser.parse_args()
    
    print("Text-Guided ReID Evaluation")
    print(f"Model weights: {args.model_weights}")
    print(f"Query: {args.query_annotation}")
    print(f"Gallery: {args.gallery_annotation}")
    print()
    
    # TODO: 加载配置、模型和数据
    # 这里需要根据实际的模型结构来实现
    
    print("请根据实际模型结构完成评估脚本的实现")
    print("主要步骤:")
    print("1. 加载配置 (如果使用)")
    print("2. 构建模型并加载权重")
    print("3. 创建数据加载器 (使用 TextReIDDataset)")
    print("4. 调用 evaluate_text_to_image 函数")


if __name__ == '__main__':
    main()

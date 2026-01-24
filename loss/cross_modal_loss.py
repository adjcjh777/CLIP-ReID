# loss/cross_modal_loss.py
"""
Cross-Modal Loss Functions for Text-Guided ReID

主要包含:
1. CrossModalContrastiveLoss: 跨模态对比学习损失 (InfoNCE)
2. TextImageMatchingLoss: 综合文本-图像匹配损失
3. CMPMLoss: Cross-Modal Projection Matching Loss
4. TextImageTripletLoss: 跨模态三元组损失

Reference:
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- CMPM: Deep Cross-Modal Projection Learning (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CrossModalContrastiveLoss(nn.Module):
    """
    跨模态对比学习损失 (InfoNCE)
    
    基于 CLIP 的对比学习目标，同时优化:
    - Text-to-Image (T2I): 给定文本，找到对应的图像
    - Image-to-Text (I2T): 给定图像，找到对应的文本
    
    与标准 CLIP 的区别：
    - 支持同一身份的多个正样本（ReID 场景）
    - 可选的标签平滑
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        symmetric: bool = True,
        learnable_temperature: bool = False
    ):
        """
        Args:
            temperature: 温度参数，控制分布的锐度
            label_smoothing: 标签平滑参数
            symmetric: 是否使用对称损失 (T2I + I2T)
            learnable_temperature: 是否学习温度参数
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.symmetric = symmetric
        
        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
            )
        else:
            self.register_buffer(
                'logit_scale',
                torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
            )
    
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
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * text_features @ image_features.T  # [B, B]
        
        # 构建正样本 mask（同一身份的为正样本）
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()  # [B, B]
        
        # 确保对角线是正样本
        eye_mask = torch.eye(batch_size, device=device)
        pos_mask = torch.max(pos_mask, eye_mask)
        
        # 计算每个样本的正样本数量
        num_pos = pos_mask.sum(dim=1)  # [B]
        
        # Text-to-Image Loss
        t2i_loss = self._compute_loss(logits, pos_mask, num_pos)
        
        if self.symmetric:
            # Image-to-Text Loss
            i2t_loss = self._compute_loss(logits.T, pos_mask.T, num_pos)
            loss = (t2i_loss + i2t_loss) / 2
        else:
            loss = t2i_loss
        
        return loss
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        num_pos: torch.Tensor
    ) -> torch.Tensor:
        """计算单向对比损失"""
        # 数值稳定性
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 计算 log softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # 只计算正样本的损失
        loss = -(pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)
        
        return loss.mean()


class TextImageTripletLoss(nn.Module):
    """
    跨模态三元组损失
    
    对于每个文本 query:
    - Anchor: 文本特征
    - Positive: 同一身份的图像特征
    - Negative: 不同身份的图像特征
    """
    
    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        """
        Args:
            margin: 间隔参数
            hard_mining: 是否使用困难样本挖掘
        """
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算跨模态三元组损失
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # L2 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算距离矩阵（1 - 余弦相似度）
        similarity = text_features @ image_features.T  # [B, B]
        distance = 1 - similarity
        
        # 构建 mask
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()  # [B, B]
        neg_mask = 1 - pos_mask
        
        if self.hard_mining:
            # 困难正样本：距离最大的正样本
            pos_dist = distance * pos_mask + (-1e9) * neg_mask
            hardest_pos, _ = pos_dist.max(dim=1)  # [B]
            
            # 困难负样本：距离最小的负样本
            neg_dist = distance * neg_mask + 1e9 * pos_mask
            hardest_neg, _ = neg_dist.min(dim=1)  # [B]
        else:
            # 随机采样
            pos_indices = torch.multinomial(pos_mask + 1e-8, 1).squeeze()
            neg_indices = torch.multinomial(neg_mask + 1e-8, 1).squeeze()
            
            hardest_pos = distance[torch.arange(batch_size), pos_indices]
            hardest_neg = distance[torch.arange(batch_size), neg_indices]
        
        # Triplet Margin Loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin).mean()
        
        return loss


class CMPMLoss(nn.Module):
    """
    Cross-Modal Projection Matching (CMPM) Loss
    
    参考: Deep Cross-Modal Projection Learning (ECCV 2018)
    
    使用 KL 散度来优化跨模态分布匹配
    """
    
    def __init__(self, epsilon: float = 1e-8, lambda_: float = 1.0):
        """
        Args:
            epsilon: 数值稳定性参数
            lambda_: 损失权重
        """
        super().__init__()
        self.epsilon = epsilon
        self.lambda_ = lambda_
    
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
        
        # 计算相似度
        similarity = text_features @ image_features.T  # [B, B]
        
        # 计算概率分布
        p_t2i = F.softmax(similarity, dim=1)  # [B, B]
        p_i2t = F.softmax(similarity.T, dim=1)  # [B, B]
        
        # 目标分布：同一身份的均匀分布
        num_pos = label_mask.sum(dim=1, keepdim=True)
        target = label_mask / (num_pos + self.epsilon)
        
        # KL 散度
        kl_t2i = (target * (torch.log(target + self.epsilon) - 
                           torch.log(p_t2i + self.epsilon))).sum(dim=1).mean()
        kl_i2t = (target.T * (torch.log(target.T + self.epsilon) - 
                             torch.log(p_i2t + self.epsilon))).sum(dim=1).mean()
        
        return self.lambda_ * (kl_t2i + kl_i2t) / 2


class TextImageMatchingLoss(nn.Module):
    """
    综合文本-图像匹配损失
    
    结合多种损失函数:
    1. 跨模态对比损失 (InfoNCE)
    2. 跨模态三元组损失
    3. CMPM 损失
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.3,
        contrastive_weight: float = 1.0,
        triplet_weight: float = 0.5,
        cmpm_weight: float = 0.5
    ):
        """
        Args:
            temperature: 温度参数
            margin: 三元组损失的间隔
            contrastive_weight: 对比损失权重
            triplet_weight: 三元组损失权重
            cmpm_weight: CMPM 损失权重
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        self.cmpm_weight = cmpm_weight
        
        # 各个损失函数
        self.contrastive_loss = CrossModalContrastiveLoss(
            temperature=temperature,
            symmetric=True,
            learnable_temperature=True
        )
        
        self.triplet_loss = TextImageTripletLoss(
            margin=margin,
            hard_mining=True
        )
        
        self.cmpm_loss = CMPMLoss()
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]
            labels: 身份标签 [B]
        
        Returns:
            dict: 包含各项损失和总损失
        """
        losses = {}
        total_loss = 0
        
        # 对比损失
        if self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(
                text_features, image_features, labels
            )
            losses['contrastive'] = contrastive
            total_loss = total_loss + self.contrastive_weight * contrastive
        
        # 三元组损失
        if self.triplet_weight > 0:
            triplet = self.triplet_loss(
                text_features, image_features, labels
            )
            losses['triplet'] = triplet
            total_loss = total_loss + self.triplet_weight * triplet
        
        # CMPM 损失
        if self.cmpm_weight > 0:
            cmpm = self.cmpm_loss(
                text_features, image_features, labels
            )
            losses['cmpm'] = cmpm
            total_loss = total_loss + self.cmpm_weight * cmpm
        
        losses['total'] = total_loss
        
        return losses


def make_text_reid_loss(cfg) -> TextImageMatchingLoss:
    """
    创建 Text-ReID 损失函数
    
    Args:
        cfg: 配置对象
    
    Returns:
        TextImageMatchingLoss 实例
    """
    temperature = getattr(cfg.MODEL, 'TEXT_TEMPERATURE', 0.07)
    margin = getattr(cfg.SOLVER, 'MARGIN', 0.3)
    contrastive_weight = getattr(cfg.MODEL, 'CONTRASTIVE_LOSS_WEIGHT', 1.0)
    triplet_weight = getattr(cfg.MODEL, 'TEXT_TRIPLET_LOSS_WEIGHT', 0.5)
    cmpm_weight = getattr(cfg.MODEL, 'CMPM_LOSS_WEIGHT', 0.5)
    
    return TextImageMatchingLoss(
        temperature=temperature,
        margin=margin,
        contrastive_weight=contrastive_weight,
        triplet_weight=triplet_weight,
        cmpm_weight=cmpm_weight
    )

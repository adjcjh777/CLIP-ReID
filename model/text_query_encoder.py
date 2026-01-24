# model/text_query_encoder.py
"""
Text Query Encoder for Text-Guided ReID

基于 CLIP Text Encoder，添加可学习的查询优化模块，实现：
1. Query Refinement: 优化查询表示
2. Attribute Attention: 关注关键属性词
3. Cross-Modal Matching: 与图像特征匹配

Usage:
    from model.text_query_encoder import TextQueryEncoder, CrossModalMatcher
    
    # 创建编码器
    text_encoder = TextQueryEncoder(clip_model)
    
    # 编码文本
    text_features = text_encoder(tokenized_text)
    
    # 跨模态匹配
    matcher = CrossModalMatcher()
    similarity = matcher(text_features, image_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union


class QueryRefinementBlock(nn.Module):
    """
    查询优化块
    
    使用残差连接和多层感知机优化查询表示
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个残差连接
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        
        # 最终归一化
        x = self.norm2(x)
        
        return x


class AttributeAttention(nn.Module):
    """
    属性注意力模块
    
    使用可学习的属性 queries 来提取查询中的关键属性信息
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_attributes: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 特征维度
            num_attributes: 属性数量（如性别、年龄、上衣颜色等）
            num_heads: 注意力头数
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.num_attributes = num_attributes
        
        # 可学习的属性 queries
        # 对应属性: gender, age, hair, upper_color, upper_style, 
        #          lower_color, lower_style, accessories
        self.attribute_queries = nn.Parameter(
            torch.randn(num_attributes, embed_dim) * 0.02
        )
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self, 
        text_features: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            text_features: 文本特征 [B, D]
            return_attention: 是否返回注意力权重
        
        Returns:
            增强的文本特征 [B, D]
            （可选）注意力权重 [B, num_attributes, 1]
        """
        batch_size = text_features.shape[0]
        
        # 扩展属性 queries
        attr_queries = self.attribute_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_attr, D]
        
        # 将文本特征扩展为 key/value
        text_kv = text_features.unsqueeze(1)  # [B, 1, D]
        
        # 注意力计算
        attr_features, attn_weights = self.attention(
            query=attr_queries,
            key=text_kv,
            value=text_kv,
            need_weights=True
        )  # attr_features: [B, num_attr, D], attn_weights: [B, num_attr, 1]
        
        # 池化属性特征
        attr_pooled = attr_features.mean(dim=1)  # [B, D]
        
        # 融合原始特征和属性特征
        concat_features = torch.cat([text_features, attr_pooled], dim=-1)
        enhanced_features = self.fusion(concat_features)
        
        if return_attention:
            return enhanced_features, attn_weights
        
        return enhanced_features


class TextQueryEncoder(nn.Module):
    """
    文本查询编码器
    
    将自然语言描述编码为特征向量，用于与图像特征匹配。
    
    架构:
    1. CLIP Text Encoder: 将文本编码为初始特征
    2. Query Refinement: 使用 MLP 优化查询表示
    3. Attribute Attention: 使用注意力机制提取关键属性
    4. Output Projection: 最终特征投影
    """
    
    def __init__(
        self,
        clip_model,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_refinement_layers: int = 2,
        num_attributes: int = 8,
        dropout: float = 0.1,
        use_attribute_attention: bool = True
    ):
        """
        Args:
            clip_model: CLIP 模型实例
            embed_dim: 输出特征维度
            hidden_dim: 隐藏层维度
            num_refinement_layers: Query Refinement 层数
            num_attributes: 属性数量
            dropout: Dropout 比例
            use_attribute_attention: 是否使用属性注意力
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
        self.refinement_layers = nn.ModuleList([
            QueryRefinementBlock(embed_dim, hidden_dim, dropout)
            for _ in range(num_refinement_layers)
        ])
        
        # 属性注意力模块
        self.use_attribute_attention = use_attribute_attention
        if use_attribute_attention:
            self.attribute_attention = AttributeAttention(
                embed_dim=embed_dim,
                num_attributes=num_attributes,
                dropout=dropout
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def encode_text_clip(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        """
        使用 CLIP 文本编码器编码文本
        
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
        x = x[torch.arange(x.shape[0], device=x.device), 
              tokenized_text.argmax(dim=-1)] @ self.text_projection
        
        return x.float()  # 确保返回 float32
    
    def forward(
        self,
        tokenized_text: torch.Tensor,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            tokenized_text: tokenized 文本，形状 [B, L]
            return_intermediate: 是否返回中间特征
        
        Returns:
            优化后的查询特征，形状 [B, D]
            （可选）中间特征字典
        """
        # 1. CLIP 文本编码
        text_features = self.encode_text_clip(tokenized_text)
        
        intermediate = {'clip_features': text_features}
        
        # 2. Query Refinement
        refined_features = text_features
        for layer in self.refinement_layers:
            refined_features = layer(refined_features)
        
        intermediate['refined_features'] = refined_features
        
        # 3. Attribute Attention
        if self.use_attribute_attention:
            enhanced_features, attn_weights = self.attribute_attention(
                refined_features, return_attention=True
            )
            intermediate['attr_features'] = enhanced_features
            intermediate['attr_attention'] = attn_weights
        else:
            enhanced_features = refined_features
        
        # 4. 输出投影
        output_features = self.output_projection(enhanced_features)
        
        # 5. L2 归一化
        output_features = F.normalize(output_features, p=2, dim=-1)
        
        intermediate['output_features'] = output_features
        
        if return_intermediate:
            return intermediate
        
        return output_features


class CrossModalMatcher(nn.Module):
    """
    跨模态匹配模块
    
    计算文本查询和图像特征之间的相似度，支持：
    1. 余弦相似度
    2. 可学习的温度参数
    3. 多种相似度计算方式
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        """
        Args:
            embed_dim: 特征维度
            temperature: 温度参数
            learnable_temperature: 是否学习温度参数
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        if learnable_temperature:
            # 初始化为 log(1/temperature)
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
        normalize: bool = True
    ) -> torch.Tensor:
        """
        计算文本与图像的匹配分数
        
        Args:
            text_features: 文本特征 [B_t, D] 或 [1, D]
            image_features: 图像特征 [B_i, D]
            normalize: 是否进行 L2 归一化
        
        Returns:
            相似度矩阵 [B_t, B_i]
        """
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * text_features @ image_features.T
        
        return similarity
    
    def get_temperature(self) -> float:
        """获取当前温度值"""
        return 1.0 / self.logit_scale.exp().item()


def build_text_query_encoder(clip_model, cfg=None) -> TextQueryEncoder:
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
        hidden_dim = getattr(cfg.MODEL, 'TEXT_HIDDEN_DIM', 1024)
        num_layers = getattr(cfg.MODEL, 'TEXT_NUM_LAYERS', 2)
        num_attributes = getattr(cfg.MODEL, 'TEXT_NUM_ATTRIBUTES', 8)
        dropout = getattr(cfg.MODEL, 'TEXT_DROPOUT', 0.1)
        use_attr_attn = getattr(cfg.MODEL, 'USE_ATTRIBUTE_ATTENTION', True)
    else:
        embed_dim = 512
        hidden_dim = 1024
        num_layers = 2
        num_attributes = 8
        dropout = 0.1
        use_attr_attn = True
    
    return TextQueryEncoder(
        clip_model=clip_model,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_refinement_layers=num_layers,
        num_attributes=num_attributes,
        dropout=dropout,
        use_attribute_attention=use_attr_attn
    )


class TextImageReIDModel(nn.Module):
    """
    完整的 Text-Image ReID 模型
    
    整合图像编码器和文本编码器，实现端到端的训练和推理
    """
    
    def __init__(
        self,
        clip_model,
        num_classes: int,
        embed_dim: int = 512
    ):
        """
        Args:
            clip_model: CLIP 模型
            num_classes: 身份类别数
            embed_dim: 特征维度
        """
        super().__init__()
        
        # 图像编码器
        self.image_encoder = clip_model.visual
        
        # 文本编码器
        self.text_encoder = TextQueryEncoder(clip_model, embed_dim=embed_dim)
        
        # 跨模态匹配器
        self.matcher = CrossModalMatcher(embed_dim=embed_dim)
        
        # 图像分类器（用于 ID loss）
        self.image_classifier = nn.Linear(embed_dim, num_classes)
        
        # BN layer
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        features = self.image_encoder(images)
        if hasattr(features, 'shape') and len(features.shape) == 3:
            features = features[:, 0]  # 取 CLS token
        features = F.normalize(features, p=2, dim=-1)
        return features
    
    def encode_text(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.text_encoder(tokenized_text)
    
    def forward(
        self,
        images: torch.Tensor = None,
        texts: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 图像 [B, 3, H, W]
            texts: tokenized 文本 [B, L]
            labels: 身份标签 [B]
        
        Returns:
            包含各种特征和分数的字典
        """
        outputs = {}
        
        if images is not None:
            image_features = self.encode_image(images)
            outputs['image_features'] = image_features
            
            # BN 和分类
            bn_features = self.bottleneck(image_features)
            outputs['image_bn_features'] = bn_features
            
            if self.training:
                outputs['image_logits'] = self.image_classifier(bn_features)
        
        if texts is not None:
            text_features = self.encode_text(texts)
            outputs['text_features'] = text_features
        
        # 计算跨模态相似度
        if 'image_features' in outputs and 'text_features' in outputs:
            similarity = self.matcher(
                outputs['text_features'],
                outputs['image_features']
            )
            outputs['similarity'] = similarity
        
        return outputs

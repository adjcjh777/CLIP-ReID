# model/text_query_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextQueryEncoder(nn.Module):
    """
    基于 CLIP Text Encoder 的查询编码器
    """
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        # 属性注意力模块 (简化版)
        self.attr_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.query_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, text, return_all=False):
        """
        text: [B, 77] tokens
        """
        x = self.token_embedding(text).type(self.dtype)  # [B, 77, 512]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # [EOS] token 特征 (原始 CLIP)
        # x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        # 使用注意力聚合特征
        # Query: learnable query, Key/Value: text features
        B = x.shape[0]
        query = self.query_embed.expand(B, -1, -1) # [B, 1, D]
        feat, _ = self.attr_attention(query, x.float(), x.float()) # [B, 1, D]
        feat = feat.squeeze(1)
        
        return feat

def build_text_encoder(clip_model):
    return TextQueryEncoder(clip_model)

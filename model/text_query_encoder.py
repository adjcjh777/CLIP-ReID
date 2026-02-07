# model/text_query_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextQueryEncoder(nn.Module):
    """
    基于 CLIP Text Encoder 的查询编码器
    
    使用 CLIP Transformer 编码文本，然后通过属性注意力模块
    对所有 token 特征进行聚合，产出与 CLIP text projection 
    同维度 (512) 的特征。
    """
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        # Transformer hidden dim (same as positional_embedding's last dim)
        transformer_width = self.positional_embedding.shape[-1]
        
        # 属性注意力模块
        self.attr_attention = nn.MultiheadAttention(transformer_width, num_heads=8, batch_first=True)
        self.query_embed = nn.Parameter(torch.randn(1, 1, transformer_width) * 0.02)
        
        # Project from transformer_width to embed_dim (same as text_projection output)
        if transformer_width != embed_dim:
            self.output_proj = nn.Linear(transformer_width, embed_dim, bias=False)
        else:
            self.output_proj = None
        
    def forward(self, text, return_all=False):
        """
        text: [B, 77] tokens from clip.tokenize
        Returns: [B, embed_dim] text features
        """
        x = self.token_embedding(text).type(self.dtype)  # [B, 77, transformer_width]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 使用注意力聚合特征
        B = x.shape[0]
        query = self.query_embed.expand(B, -1, -1).float()  # [B, 1, D]
        feat, _ = self.attr_attention(query, x.float(), x.float())  # [B, 1, D]
        feat = feat.squeeze(1)  # [B, D]
        
        # Project to output dimension
        if self.output_proj is not None:
            feat = self.output_proj(feat)
        else:
            # Use CLIP's text_projection
            feat = feat @ self.text_projection
        
        return feat

def build_text_encoder(clip_model, embed_dim=512):
    return TextQueryEncoder(clip_model, embed_dim=embed_dim)

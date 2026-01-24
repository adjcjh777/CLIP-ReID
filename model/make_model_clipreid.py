import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MultiGranularityHead(nn.Module):
    """
    多粒度特征提取头
    
    将 ViT 的 Patch tokens 水平划分为 num_parts 个部分，
    对每个部分进行 GAP + BNNeck + Classifier 处理。
    
    Args:
        in_planes: 输入特征维度 (ViT-B/16 = 768)
        num_classes: 训练集身份类别数
        num_parts: 划分的部分数量，默认 4
        reduce_dim: 测试时降维的目标维度，默认 512
    """
    def __init__(self, in_planes, num_classes, num_parts=4, reduce_dim=512):
        super().__init__()
        self.num_parts = num_parts
        self.in_planes = in_planes
        self.reduce_dim_size = reduce_dim
        
        # 每个 Part 一个 BNNeck
        self.bottlenecks = nn.ModuleList()
        for _ in range(num_parts):
            bn = nn.BatchNorm1d(in_planes)
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
            self.bottlenecks.append(bn)
        
        # 每个 Part 一个分类器
        self.classifiers = nn.ModuleList()
        for _ in range(num_parts):
            fc = nn.Linear(in_planes, num_classes, bias=False)
            fc.apply(weights_init_classifier)
            self.classifiers.append(fc)
        
        # 降维层
        self.reduce_layers = nn.ModuleList()
        for _ in range(num_parts):
            layer = nn.Linear(in_planes, reduce_dim)
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
            nn.init.constant_(layer.bias, 0.0)
            self.reduce_layers.append(layer)
        
    def forward(self, patch_features):
        """
        Args:
            patch_features: [B, H*W, C] 不含 CLS token
        Returns:
            训练: (part_scores, part_feats)
            测试: concat_feat
        """
        B, N, C = patch_features.shape
        W = 8  # 假设宽度 128/16=8
        H = N // W
        
        # Reshape: [B, N, C] -> [B, H, W, C]
        patch_features = patch_features.view(B, H, W, C)
        
        # 水平划分
        part_size = H // self.num_parts
        parts = []
        for i in range(self.num_parts):
            part = patch_features[:, i*part_size:(i+1)*part_size, :, :]
            part = part.mean(dim=[1, 2])  # GAP: [B, C]
            parts.append(part)
        
        # 处理每个 part
        part_features = []
        part_scores = []
        for i, part in enumerate(parts):
            feat = self.bottlenecks[i](part)
            part_features.append(feat)
            if self.training:
                score = self.classifiers[i](feat)
                part_scores.append(score)
        
        if self.training:
            return part_scores, part_features
        else:
            reduced = [self.reduce_layers[i](f) for i, f in enumerate(part_features)]
            return torch.cat(reduced, dim=1)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

        # 多粒度特征模块
        self.multi_granularity_enabled = getattr(cfg.MODEL, 'MULTI_GRANULARITY', None) is not None and \
                                          getattr(cfg.MODEL.MULTI_GRANULARITY, 'ENABLED', False)
        if self.multi_granularity_enabled:
            num_parts = cfg.MODEL.MULTI_GRANULARITY.NUM_PARTS
            part_dim = cfg.MODEL.MULTI_GRANULARITY.PART_DIM
            self.multi_granularity_head = MultiGranularityHead(
                in_planes=self.in_planes,
                num_classes=num_classes,
                num_parts=num_parts,
                reduce_dim=part_dim
            )
            print(f'Multi-Granularity enabled: {num_parts} parts, reduce_dim={part_dim}')

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

            # 多粒度特征提取
            part_scores = None
            part_feats = None
            if self.multi_granularity_enabled:
                patch_tokens = image_features[:, 1:]  # 去掉 CLS token
                if self.training:
                    part_scores, part_feats = self.multi_granularity_head(patch_tokens)
                else:
                    part_concat_feat = self.multi_granularity_head(patch_tokens)

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            if self.model_name == 'ViT-B-16' and self.multi_granularity_enabled:
                return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj, part_scores, part_feats
            else:
                return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                global_feat = torch.cat([feat, feat_proj], dim=1)
            else:
                global_feat = torch.cat([img_feature, img_feature_proj], dim=1)
            
            if self.model_name == 'ViT-B-16' and self.multi_granularity_enabled:
                return torch.cat([global_feat, part_concat_feat], dim=1)
            else:
                return global_feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 


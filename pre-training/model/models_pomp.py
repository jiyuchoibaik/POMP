from functools import partial
import torch
import torch.nn as nn
import numpy as np
import math
import timm.models.vision_transformer
from timm.models.vision_transformer import Block

num_omics = 3


class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.scale_factor = 1.0 / (input_dim ** 0.5)

    def forward(self, query, key, value):
        query_proj = self.query_linear(query)  # Query的线性变换
        key_proj = self.key_linear(key)  # Key的线性变换
        value_proj = self.value_linear(value)  # Value的线性变换

        attention_scores = torch.matmul(query_proj, key_proj.transpose(-1, -2))  # 计算注意力分数
        attention_scores = attention_scores * self.scale_factor # 缩放点积
        attention_weights = torch.softmax(attention_scores, dim=-1)  # 计算注意力权重

        attended_values = torch.matmul(attention_weights, value_proj)  # 对Value加权求和

        return attended_values


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[0, :, :]


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # self.patch_embed = PatchEmbed(self.img_size, self.patch_size, in_chans, self.embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_omics, self.embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.linear = nn.Linear(300, self.embed_dim, bias=True)
        self.encoder_image = nn.Linear(1000, self.embed_dim, bias=True)

        self.pom_head = nn.Linear(self.embed_dim, 2)
        self.mom_head = nn.Linear(self.embed_dim, 300)

        self.vits = nn.ModuleList([
            Block(self.embed_dim, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.norm_vits = nn.LayerNorm(self.embed_dim)

        self.img_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.norm_img_transf = nn.LayerNorm(self.embed_dim)

        self.omics_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.norm_omics_transf = nn.LayerNorm(self.embed_dim)

        self.fuse_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(2)])
        self.norm_fuse_transf = nn.LayerNorm(self.embed_dim)

        self.cross_attn = nn.ModuleList([
            CrossAttention(self.embed_dim)
            for i in range(1)])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_features(self, samples):
        regions, X_mrna, X_mirna, X_meth = samples[:]
        regions = regions.squeeze(0)

        # embed patches
        try:
            reg_emb = self.patch_embed(regions)
        except Exception as e:
            print(e.args)
            print("shape: ", regions.shape)

        # ViT encoding --> patch embedding
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits:
            reg_emb = blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img = torch.mean(reg_emb, dim=1)

        # add cls token, and combine position enbedding
        img = torch.cat((self.cls_token, img), dim=0)
        positional_encoding = PositionalEncoding(self.embed_dim, img.shape[0]).to(img.device)
        img = positional_encoding(img)
        img = img.unsqueeze(0)

        # Image-Transformer
        img = self.pos_drop(img)
        for blk in self.img_transf:
            img = blk(img)
        img = self.norm_img_transf(img)
        img_cls = img[0, 0:1, :]

        # append omics token
        X_mrna = self.linear(X_mrna).unsqueeze(1)
        X_mirna = self.linear(X_mirna).unsqueeze(1)
        X_meth = self.linear(X_meth).unsqueeze(1)
        X_omics = torch.cat((self.cls_token.unsqueeze(1), X_mrna, X_mirna, X_meth), dim=1)

        X_omics = self.pos_drop(X_omics)
        for blk in self.omics_transf:
            X_omics = blk(X_omics)
        X_omics = self.norm_omics_transf(X_omics)

        omics_cls = X_omics[0, 0:1, :]

        return img_cls, omics_cls, img, X_omics

    def path_guided_omics_encoder(self, image_embeds, omics_embeds, idx_mask):

        omics_embeds = self.pos_drop(omics_embeds)
        for blk in self.cross_attn:
            omics_embeds = blk(query=omics_embeds, key=image_embeds, value=image_embeds)

        for blk in self.fuse_transf:
            omics_embeds = blk(omics_embeds)

        path_guid_omics_embeds = self.norm_fuse_transf(omics_embeds)

        logit_cls = self.pom_head(path_guid_omics_embeds[:, 0, :])
        logit_mask = self.mom_head(path_guid_omics_embeds[:, idx_mask, :])

        return logit_mask, logit_cls

    def forward(self, x):
        img_cls, omics_cls, image_embed, omics_embed = self.forward_features(x)

        return img_cls, omics_cls, image_embed, omics_embed



def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=384, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

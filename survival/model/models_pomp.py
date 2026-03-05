from functools import partial
import torch
import torch.nn as nn
import numpy as np
import math
import timm.models.vision_transformer
from timm.models.vision_transformer import Block

# 원본: num_omics = 3 (mRNA, miRNA, meth)
# 수정: num_omics = 1 (RNA-seq)
num_omics = 1


class GlobalAttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_in  = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, x):
        hidden     = torch.relu(self.linear_in(x))
        attn       = self.linear_out(hidden).squeeze(2)
        weights    = self.softmax(attn)
        context    = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, query, key, value):
        attn = torch.matmul(self.q(query), self.k(key).transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, self.v(value))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe  = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[0, :x.shape[0], :]


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    원본 대비 변경:
      - num_omics: 3 → 1
      - self.linear: nn.Linear(300, embed_dim) → nn.Linear(n_genes, embed_dim)
      - forward_features: X_mrna/mirna/meth 3개 → X_rna 1개
      - path_guided_omics_encoder: 구조 동일, omics 차원만 다름
    """

    def __init__(self, global_pool=False, n_genes=2000, **kwargs):
        super().__init__(**kwargs)

        num_patches    = self.patch_embed.num_patches
        self.n_genes   = n_genes

        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        # pos_embed: cls(1) + patches + omics(num_omics=1)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1 + num_omics, self.embed_dim),
            requires_grad=False
        )

        # 원본: nn.Linear(300, embed_dim)  →  nn.Linear(n_genes, embed_dim)
        self.linear      = nn.Linear(n_genes, self.embed_dim, bias=True)
        self.risk_head   = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Sigmoid())
        self.gap         = GlobalAttentionPooling(self.embed_dim, self.embed_dim)

        self.vits = nn.ModuleList([
            Block(self.embed_dim, num_heads=6, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)])
        self.norm_vits = nn.LayerNorm(self.embed_dim)

        self.img_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)])
        self.norm_img_transf = nn.LayerNorm(self.embed_dim)

        self.omics_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)])
        self.norm_omics_transf = nn.LayerNorm(self.embed_dim)

        self.fuse_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)])
        self.norm_fuse_transf = nn.LayerNorm(self.embed_dim)

        self.cross_attn = nn.ModuleList([CrossAttention(self.embed_dim)])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_features(self, samples):
        """
        Args:
            samples: [regions, X_rna]
              regions: (1, N_patch, 3, 256, 256)
              X_rna  : (1, n_genes)
        """
        regions, X_rna = samples
        regions = regions.squeeze(0)   # (N_patch, 3, 256, 256)

        # ── Patch embedding + ViT ────────────────────────────────────────
        reg_emb = self.patch_embed(regions)
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits:
            reg_emb = blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img = torch.mean(reg_emb, dim=1)  # (N_patch, D)

        # ── Image-Transformer ────────────────────────────────────────────
        img = torch.cat((self.cls_token, img), dim=0)  # (N_patch+1, D)
        pe  = PositionalEncoding(self.embed_dim, img.shape[0]).to(img.device)
        img = pe(img).unsqueeze(0)                     # (1, N_patch+1, D)

        img = self.pos_drop(img)
        for blk in self.img_transf:
            img = blk(img)
        img     = self.norm_img_transf(img)
        img_cls = img[0, 0:1, :]                       # (1, 1, D)

        # ── Omics Transformer (RNA-seq only) ─────────────────────────────
        X_rna_emb = self.linear(X_rna).unsqueeze(1)   # (1, 1, D)
        # cls + 1 omics token
        X_omics = torch.cat((self.cls_token.unsqueeze(1), X_rna_emb), dim=1)  # (1, 2, D)

        X_omics = self.pos_drop(X_omics)
        for blk in self.omics_transf:
            X_omics = blk(X_omics)
        X_omics   = self.norm_omics_transf(X_omics)
        omics_cls = X_omics[0, 0:1, :]                # (1, 1, D)

        # cosine similarity (원본과 동일)
        corr = torch.nn.functional.cosine_similarity(img_cls, omics_cls)

        return corr, img, X_omics

    def path_guided_omics_encoder(self, image_embeds, omics_embeds0):
        omics_embeds = self.pos_drop(omics_embeds0)
        for blk in self.cross_attn:
            omics_embeds = blk(query=omics_embeds,
                               key=image_embeds, value=image_embeds)
        for blk in self.fuse_transf:
            omics_embeds = blk(omics_embeds)

        # residual
        omics_embeds = omics_embeds + omics_embeds0
        path_guid    = self.norm_fuse_transf(omics_embeds)

        logit_cls = self.risk_head(path_guid[:, 0:1, :])
        return logit_cls[0]

    def forward(self, x):
        corr, image_embed, omics_embed = self.forward_features(x)
        return corr, image_embed, omics_embed


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=384, depth=6,
        num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
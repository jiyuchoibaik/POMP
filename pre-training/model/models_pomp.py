# models_pomp.py  ─ TCGA-LUAD (RNA-seq 단일 omics)
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import Block

NUM_OMICS = 1


class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q  = nn.Linear(dim, dim)
        self.k  = nn.Linear(dim, dim)
        self.v  = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, query, key, value):
        attn = torch.matmul(self.q(query), self.k(key).transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, self.v(value))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[0, :x.size(1), :]


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, rna_dim: int = 2000, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.rna_dim  = rna_dim
        num_patches   = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1 + NUM_OMICS, self.embed_dim),
            requires_grad=False
        )

        self.rna_linear = nn.Linear(rna_dim, self.embed_dim, bias=True)
        self.pom_head   = nn.Linear(self.embed_dim, 2)
        self.mom_head   = nn.Linear(self.embed_dim, rna_dim)

        self.vits = nn.ModuleList([
            Block(self.embed_dim, num_heads=6, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_vits = nn.LayerNorm(self.embed_dim)

        self.img_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_img_transf = nn.LayerNorm(self.embed_dim)

        self.omics_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_omics_transf = nn.LayerNorm(self.embed_dim)

        self.fuse_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_fuse_transf = nn.LayerNorm(self.embed_dim)

        self.cross_attn = nn.ModuleList([CrossAttention(self.embed_dim)])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_features(self, samples):
        """
        Args:
            regions : (B, N_patches, 3, 256, 256)
            x_rna   : (B, rna_dim)
        Returns:
            img_cls    : (B, 1, D)
            omics_cls  : (B, 1, D)
            img        : (B, N+1, D)
            omics_inp  : (B, 2, D)
        """
        regions, x_rna = samples
        B = x_rna.shape[0]
        N = regions.shape[1]

        # (B, N, 3, 256, 256) → (B*N, 3, 256, 256)
        regions = regions.view(B * N, 3, 256, 256)

        # ── 1. Patch embedding + ViT ─────────────────────────────────────
        reg_emb = self.patch_embed(regions)          # (B*N, n_tokens, D)
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits:
            reg_emb = blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img = torch.mean(reg_emb, dim=1)             # (B*N, D)
        img = img.view(B, N, self.embed_dim)         # (B, N, D)

        # ── 2. Image-level Transformer ───────────────────────────────────
        cls_tokens = self.cls_token.unsqueeze(0).expand(B, 1, -1)  # (B, 1, D)
        img = torch.cat([cls_tokens, img], dim=1)    # (B, N+1, D)
        pe  = PositionalEncoding(self.embed_dim, img.shape[1]).to(img.device)
        img = pe(img)

        img = self.pos_drop(img)
        for blk in self.img_transf:
            img = blk(img)
        img     = self.norm_img_transf(img)
        img_cls = img[:, 0:1, :]                     # (B, 1, D)

        # ── 3. Omics Transformer ─────────────────────────────────────────
        rna_emb   = self.rna_linear(x_rna).unsqueeze(1)            # (B, 1, D)
        omics_inp = torch.cat([cls_tokens, rna_emb], dim=1)        # (B, 2, D)

        omics_inp = self.pos_drop(omics_inp)
        for blk in self.omics_transf:
            omics_inp = blk(omics_inp)
        omics_inp = self.norm_omics_transf(omics_inp)
        omics_cls = omics_inp[:, 0:1, :]             # (B, 1, D)

        return img_cls, omics_cls, img, omics_inp

    def path_guided_omics_encoder(self, image_embed, omics_embed, mask_ratio=0.3):
        """
        Args:
            image_embed  : (1, seq, D)
            omics_embed  : (1, 2, D)
        Returns:
            logit_mask : (1, rna_dim)
            logit_cls  : (1, 2)
        """
        fused = omics_embed
        for blk in self.cross_attn:
            fused = blk(query=fused, key=image_embed, value=image_embed)

        for blk in self.fuse_transf:
            fused = blk(fused)
        fused = self.norm_fuse_transf(fused)

        logit_cls  = self.pom_head(fused[:, 0, :])
        logit_mask = self.mom_head(fused[:, 1, :])

        return logit_mask, logit_cls

    def forward(self, x):
        img_cls, omics_cls, img_embed, omics_embed = self.forward_features(x)
        return img_cls, omics_cls, img_embed, omics_embed


def vit_base_patch16(rna_dim: int = 2000, **kwargs):
    model = VisionTransformer(
        rna_dim    = rna_dim,
        img_size   = 256,
        patch_size = 16,
        embed_dim  = 384,
        depth      = 6,
        num_heads  = 12,
        mlp_ratio  = 4,
        qkv_bias   = True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
# survival/model/models_pomp.py
# ---------------------------------------------------------------------------
# [pre-training과 체크포인트 호환] pre-training/model/models_pomp.py 와 백본 동일.
# - [수정1] forward: cosine_similarity 완전 제거 → img_risk 반환
# - [수정2] path_guided_omics_encoder: residual connection 추가 (원본 논문 구조)
# - [수정3] risk_head: Sigmoid 제거 → Cox loss는 순서만 보므로 범위 제한 불필요
# ---------------------------------------------------------------------------
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint

NUM_OMICS = 1


class CrossAttention(nn.Module):
    """pre-training과 동일. 시각화용 last_attn 저장 추가."""
    def __init__(self, dim: int):
        super().__init__()
        self.q  = nn.Linear(dim, dim)
        self.k  = nn.Linear(dim, dim)
        self.v  = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, query, key, value):
        attn = torch.matmul(self.q(query), self.k(key).transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        self.last_attn = attn.detach()  # 시각화용
        return torch.matmul(attn, self.v(value))


class PositionalEncoding(nn.Module):
    """pre-training과 동일."""
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

        # [수정3] Sigmoid 제거 → raw linear output (Cox loss는 순서만 보므로 OK)
        self.risk_head = nn.Sequential(nn.Linear(self.embed_dim, 1))
        self.gradient_checkpointing = False

    def forward_features(self, samples):
        """pre-training과 동일 (체크포인트 가중치 호환)."""
        regions, x_rna = samples
        B = x_rna.shape[0]
        N = regions.shape[1]

        regions = regions.view(B * N, 3, 256, 256)

        reg_emb = self.patch_embed(regions)
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits:
            reg_emb = checkpoint(blk, reg_emb, use_reentrant=False) if self.gradient_checkpointing else blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img = torch.mean(reg_emb, dim=1)
        img = img.view(B, N, self.embed_dim)

        cls_tokens = self.cls_token.unsqueeze(0).expand(B, 1, -1)
        img = torch.cat([cls_tokens, img], dim=1)
        pe  = PositionalEncoding(self.embed_dim, img.shape[1]).to(img.device)
        img = pe(img)

        img = self.pos_drop(img)
        for blk in self.img_transf:
            img = checkpoint(blk, img, use_reentrant=False) if self.gradient_checkpointing else blk(img)
        img     = self.norm_img_transf(img)
        img_cls = img[:, 0:1, :]

        rna_emb   = self.rna_linear(x_rna).unsqueeze(1)
        omics_inp = torch.cat([cls_tokens, rna_emb], dim=1)

        omics_inp = self.pos_drop(omics_inp)
        for blk in self.omics_transf:
            omics_inp = checkpoint(blk, omics_inp, use_reentrant=False) if self.gradient_checkpointing else blk(omics_inp)
        omics_inp = self.norm_omics_transf(omics_inp)

        # [수정1] cosine_similarity 제거 → img_cls 직접 반환
        return img_cls, img, omics_inp

    def path_guided_omics_encoder(self, image_embed, omics_embed):
        """
        [수정2] residual connection 추가 (원본 논문 구조).
        fuse_transf 통과 후 omics_embed와 residual 합산.
        """
        fused = omics_embed

        # cross-attention: RNA가 이미지 참고
        for blk in self.cross_attn:
            fused = blk(query=fused, key=image_embed, value=image_embed)

        # fuse_transf
        for blk in self.fuse_transf:
            fused = blk(fused)

        # [수정2] residual connection
        fused = fused + omics_embed

        fused = self.norm_fuse_transf(fused)

        # [수정3] Sigmoid 없이 raw linear
        risk = self.risk_head(fused[:, 0, :])  # (B, 1)
        return risk

    def forward(self, x):
        """
        [수정1] cosine_similarity 완전 제거.
        img_cls → risk_head → img_risk 반환
        engine: outputs = img_risk + path_guided_risk
        """
        img_cls, img, omics_inp = self.forward_features(x)
        # [수정3] Sigmoid 없이 raw linear
        img_risk = self.risk_head(img_cls.squeeze(1))  # (B, 1)
        return img_risk, img, omics_inp

    def get_image_cls_region_attention(self, regions):
        """pre-training과 동일 (시각화/비교 스크립트용)."""
        B, N = regions.shape[0], regions.shape[1]
        regions = regions.view(B * N, 3, 256, 256)
        reg_emb = self.patch_embed(regions)
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits:
            reg_emb = blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img = torch.mean(reg_emb, dim=1).view(B, N, self.embed_dim)
        cls_tokens = self.cls_token.unsqueeze(0).expand(B, 1, -1)
        img = torch.cat([cls_tokens, img], dim=1)
        pe = PositionalEncoding(self.embed_dim, img.shape[1]).to(img.device)
        img = pe(img)
        img = self.pos_drop(img)
        for blk in self.img_transf[:-1]:
            img = blk(img)
        last_block = self.img_transf[-1]
        x_norm = last_block.norm1(img)
        qkv = last_block.attn.qkv(x_norm)
        num_heads = getattr(last_block.attn, "num_heads", 3)
        head_dim = self.embed_dim // num_heads
        qkv = qkv.reshape(B, img.shape[1], 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = attn.detach()
        cls_to_region = attn[:, :, 0, 1:].mean(dim=1)
        return cls_to_region.cpu().float().numpy()

    def get_patch_spatial_attention(self, patch_tensor):
        """survival 시각화용: 패치 내 공간 attention."""
        with torch.no_grad():
            reg_emb = self.patch_embed(patch_tensor)
            reg_emb = self.pos_drop(reg_emb)
            for blk in self.vits[:-1]:
                reg_emb = blk(reg_emb)
            last_blk = self.vits[-1]
            x = last_blk.norm1(reg_emb)
            B, N, C = x.shape
            qkv = last_blk.attn.qkv(x)
            num_heads = last_blk.attn.num_heads
            head_dim = C // num_heads
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            importance = attn.sum(dim=1).sum(dim=1).squeeze(0)
            importance = importance.cpu().numpy().reshape(16, 16)
        return importance


def vit_base_patch16(rna_dim: int = 2000, n_genes: int = None, **kwargs):
    """pre-training과 동일. n_genes는 survival main 호환용 별칭."""
    if n_genes is not None:
        rna_dim = n_genes
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
# models_pomp.py  ─ TCGA-LUAD (RNA-seq 단일 omics)
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import Block

# omics 종류: RNA-seq 1개  (원본: mRNA+miRNA+meth 3개)
NUM_OMICS = 1


class CrossAttention(nn.Module):
    """이미지 임베딩이 omics 임베딩을 가이드하는 cross-attention"""
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
    """
    POMP 기반 멀티모달 모델 (RNA-seq 단일 omics 버전)

    변경사항 (원본 대비):
      - num_omics: 3 → 1
      - self.linear: (300 → embed_dim) → (rna_dim → embed_dim)
      - forward_features: X_mrna/X_mirna/X_meth → X_rna 단일 입력
      - path_guided_omics_encoder: gene-level masking 적용
      - pom_head / mom_head: rna_dim 기반으로 재구성
    """
    def __init__(self, rna_dim: int = 2000, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.rna_dim     = rna_dim
        num_patches      = self.patch_embed.num_patches

        # cls token & positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        # pos_embed: [1, num_patches + 1(cls) + NUM_OMICS(rna), embed_dim]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1 + NUM_OMICS, self.embed_dim),
            requires_grad=False
        )

        # RNA-seq 입력 → embed_dim 프로젝션
        self.rna_linear = nn.Linear(rna_dim, self.embed_dim, bias=True)

        # ── 예측 헤드 ──────────────────────────────────────────────────────
        # POM: Pathology-Omics Matching (매칭/비매칭 이진 분류)
        self.pom_head = nn.Linear(self.embed_dim, 2)
        # MOM: Masked Omics Modeling (마스킹된 유전자 값 재구성)
        #   마스킹 비율 30% → rna_dim * 0.3 개 재구성
        self.mom_head = nn.Linear(self.embed_dim, rna_dim)

        # ── Transformer 블록 ───────────────────────────────────────────────
        # 1) Patch-level ViT (원본 vits)
        self.vits = nn.ModuleList([
            Block(self.embed_dim, num_heads=6, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_vits = nn.LayerNorm(self.embed_dim)

        # 2) Image-level Transformer
        self.img_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_img_transf = nn.LayerNorm(self.embed_dim)

        # 3) Omics Transformer (RNA-seq)
        self.omics_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_omics_transf = nn.LayerNorm(self.embed_dim)

        # 4) Fusion Transformer (cross-attn 이후)
        self.fuse_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=3, mlp_ratio=4,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])
        self.norm_fuse_transf = nn.LayerNorm(self.embed_dim)

        # 5) Cross-Attention (image → omics)
        self.cross_attn = nn.ModuleList([CrossAttention(self.embed_dim)])

        # 온도 파라미터 (contrastive loss)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # ── Feature 추출 ──────────────────────────────────────────────────────────
    def forward_features(self, samples):
        """
        Args:
            samples: [regions, x_rna]
              regions : (N_patches, 3, 256, 256)  squeeze(0) 적용됨
              x_rna   : (1, rna_dim)
        Returns:
            img_cls   : (1, embed_dim)  이미지 cls 토큰
            omics_cls : (1, embed_dim)  omics cls 토큰
            img_embed : (1, seq, embed_dim)
            omics_embed: (1, 1+NUM_OMICS, embed_dim)
        """
        regions, x_rna = samples
        regions = regions.squeeze(0)   # (N, 3, 256, 256)

        # ── 1. Patch embedding + ViT ─────────────────────────────────────
        reg_emb = self.patch_embed(regions)          # (N, n_patch_tokens, D)
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits:
            reg_emb = blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img     = torch.mean(reg_emb, dim=1)         # (N, D)  패치 평균

        # ── 2. Image-level Transformer ───────────────────────────────────
        # cls 토큰 prepend + positional encoding
        img = torch.cat([self.cls_token, img], dim=0)      # (N+1, D)
        pe  = PositionalEncoding(self.embed_dim, img.shape[0]).to(img.device)
        img = pe(img).unsqueeze(0)                         # (1, N+1, D)

        img = self.pos_drop(img)
        for blk in self.img_transf:
            img = blk(img)
        img     = self.norm_img_transf(img)
        img_cls = img[:, 0:1, :]                           # (1, 1, D)

        # ── 3. Omics Transformer (RNA-seq) ───────────────────────────────
        rna_emb    = self.rna_linear(x_rna).unsqueeze(1)  # (1, 1, D)
        omics_inp  = torch.cat([self.cls_token.unsqueeze(1),
                                rna_emb], dim=1)           # (1, 2, D)

        omics_inp = self.pos_drop(omics_inp)
        for blk in self.omics_transf:
            omics_inp = blk(omics_inp)
        omics_inp = self.norm_omics_transf(omics_inp)
        omics_cls = omics_inp[:, 0:1, :]                   # (1, 1, D)

        return img_cls, omics_cls, img, omics_inp

    # ── Pathology-guided Omics Encoder ───────────────────────────────────────
    def path_guided_omics_encoder(self, image_embed, omics_embed, mask_ratio=0.3):
        """
        Args:
            image_embed  : (1, seq, D)
            omics_embed  : (1, 1+NUM_OMICS, D)
            mask_ratio   : gene masking 비율 (MOM 학습용)
        Returns:
            logit_mask : (1, rna_dim)  재구성된 RNA 벡터
            logit_cls  : (1, 2)        POM 분류 logit
        """
        # Cross-attention: omics가 image를 쿼리
        fused = omics_embed
        for blk in self.cross_attn:
            fused = blk(query=fused, key=image_embed, value=image_embed)

        for blk in self.fuse_transf:
            fused = blk(fused)
        fused = self.norm_fuse_transf(fused)

        logit_cls  = self.pom_head(fused[:, 0, :])    # (1, 2)  cls 토큰
        logit_mask = self.mom_head(fused[:, 1, :])    # (1, rna_dim)  RNA 토큰

        return logit_mask, logit_cls

    def forward(self, x):
        img_cls, omics_cls, img_embed, omics_embed = self.forward_features(x)
        return img_cls, omics_cls, img_embed, omics_embed


# ── 모델 생성 함수 ─────────────────────────────────────────────────────────────
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
# models_pomp_v2.py
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import timm.models.vision_transformer
from timm.models.vision_transformer import Block

NUM_OMICS = 1

'''
파이썬에서 괄호 안은 상속을 뜻함. 함수 인자가 아님.
nn.Module은 pytorch에서 모든 신경망 레이어의 *베이스 클래스*임. nn.Module을 상속하면
`parameters()`: 학습 가능한 가중치 자동 추적
`.to(device)`: GPU/CPU 이동
`.train() / eval()`: 학습/추론 모드 전환
`forward()`: 자동 호출 | model(x) 하면 forward(x) 실행
즉, `nn.Module`을 상속하지 않으면 `self.q`, `self.k`, `self.v` 같은 레이어들이 pytorch에 의해 추적되지 않아 학습이 불가
'''
class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5 # Q와 K의 내적 결과는 차원이 커질수록 그 토큰 값이 매우 커짐으로, 이에 softmax를 적용시 다른 토큰들이 0이 발생되는 경우가 있는데 이를 방지하지 위해 분산을 1로 정규화

    def forward(self, query, key, value):
        attn = torch.matmul(self.q(query), self.k(key).transpose(-1,-2)) * self.scale # 마지막 두 차원만 전치
        attn = torch.softmax(attn, dim = -1)
        return torch.matmul(attn, self.v(value))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()*-(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x+self.pe[0, :x.size(1), :] # 정수로 인덱싱하면 그 차원은 사라지므로 x.shape = (1, 10, 256)일 때 x[0]으로 슬라이싱하면 (10,256)이 됨


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, rna_dim: int = 2000, global_pool: bool = False, **kwargs): # hidden_dim, drop_out 같은 나머지 인자는 전부 **kwargs로 받음
        super().__init__(**kwargs)

        self.rna_dim = rna_dim

        # 앞서 VisionTransformer 클래스 선언시 인자로 patch_size를 받아 img_size(256x256)를 16x16으로 나누어 256개의 토큰을 만듦
        num_patches = self.patch_embed.num_patches # patch_embed는 부모 클래스인 timm의 VisionTransformer에서 자동으로 생성되어 물려받는 속성. nn.Conv2d 사용

        self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dim)) # CLS에 weight 처럼 학습되는 Paramter을 상속받아 정의하는 이유는 CLS가 입력 값을 적절히 요약한 벡터가 되어야하기 때문

        # 멀티 모달 인코더에 RNA, WSI representation이 입력으로 들어왔을 때, RNA, RNA_CLS, WSI를 구분하게 하기 위함
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patched+1+NUM_OMICS, self.embed_dim),
            requires_grad = False # 이 파라미터(weight)를 학습시키지 않기 위함. 앞서 PositionalEncoding에서 위치 정보를 계산하므로 여기서는 따로 위치 정보에 대한 학습이 필요 없음
        )

        self.rna_linear = nn.Linear(rna_dim, self.embed_dim, bias = True)
        self. omics_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=4, mlp_ratio=4,
                    qkv_bias=True, norm_layer=nn.LayerNorm) # norm_layer는 데이터 값이 explode나 vanishing하는 것을 막아주기 위한 layer
            for _ in range(2)
        ])
        self.norm_omics_transf = nn.LayerNorm(self.embed_dim)

        
        self.vits = nn.ModuleList([
            Block(self.embed_dim, num_heads = 3, mlp_ratio=4,
                    qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range (2)
        ])
        self.norm_vits = nn.LayerNorm(self.embed_dim)

        self.img_transf = nn.ModuleList([
            Block(self.embed_dim, num_head = 4, mlp_ratio =4,
                    qkv_bias = True, norm_layer = nn.LayerNorm)
            for _ in range (2)
        ])
        self.norm_img_transf = nn.LayerNorm(self.embed.dim)

        
        self.cross_attn = nn.ModuleList([CrossAttention(self.embed_dim)])
        self.fuse_transf = nn.ModuleList([
            Block(self.embed_dim, num_heads=4, mlp_ration =4,
                    qkv_bias = True, norm_layer=nn.LayerNorm)
            for _ in range (2)
        ])
        self.norm_fuse_transf = nn.LayerNorm(self.embed_dim)

        self.logit_scale = nn.Parpmeter(torch.ones([]) * np.log(1/0.07))
        self.pom_head = nn.Linear(self.embed_dim, 2)
        self.mom_head = nn.Linear(self.embed_dim, rna_dim)

    
    def forward_features(self, samples):
        regions, x_rna = samples
        B = x_rna.shape[0]
        N = regions.shape[1] # extract_patches.py에서 WSI에서 추출한 패치들

        '''
        (B, N, 3, 256, 256) -> (B*N, 3, 256, 256)
        실제 메모리 저장 순서는 [환자0패치0, 환자0패치1, ..., 환자0패치N-1, 환자1패치0, ..., 환자1패치N-1]으로 이것은 view로 shape을 바꾸면 메모리의 값과 위치는 바뀌지 않기 때문에
        원래 데이터 순서를 유지한 채 복원, 축소가 가능하여 환자별 패치들을 구별하여 학습할 수 있다.

        '''
        regions = regions.view(B*N, 3, 256, 256)

        # =====Patch embedding + ViT=====
        reg_emb = self.patch_embed(regions) #(B*N, 256(토큰 수: 256x256을 16x16으로 나눈값), 384(차원 수: embed_dim))
        reg_emb = self.pos_drop(reg_emb)
        for blk in self.vits: # vit는 마지막 두 차원에 대해 연산
            reg_emb = blk(reg_emb)
        reg_emb = self.norm_vits(reg_emb)
        img = torch.mean(reg_emb, dim = 1) # (B*N, D) 각 패치들의 평균을 내어 하나의 벡터로 압축하는 것
        img = img.view(B, N, self.embed_dim) #n (B, N, D)


        # ===== Image-level Transformer=====
        cls_tokens = self.cls_token.unsqueeze(0).expand(B, 1, -1) #(B,1,D)
        img = torch.cat([cls_tokens, img], dim = 1)
        pe = PositionalEncoding(self.embed_dim, img.shape[1]).to(img.device)
        img = pe(img)

        img = self.pos_drop(img)
        for blk in self.img_transf:
            img = blk(img)
        img = self.norm_img_transf(img)
        img_cls = img[:, 0:1, :] # (B, 1, D)

        # ===== Omics Transformer=====
        rna_emb = self.rna_linear(x_rna).unsqueeze(1) # (B, 1, D)
        omics_inp = torch.cat([cls_tokens, rna_emb], dim = 1)

        omics_inp = self.pos_drop(omics_inp)
        for blk in rna_transf:
            omics_inp = blk(omics_inp)
        omics_inp = self.norm_omics_transf(omics_inp)
        omics_cls = omics_inp[:, 0:1, :]

        return img_cls, omics_cls, img, omics_inp
    
    def path_guided_omics_encoder(self, image_embed, omics_embed, mask_ratio = 0.3):
        fused = omics_embed
        for blk in self.cross_attn:
            fused = blk(query = fused, key = image_embed, value = image_embed)
        
        for blk in self.fuse_transf:
            fused = blk(fused)
        fused = self.norm_fuse_transf(fused)

        logit_cls = self.pom_head(fused[:, 0, :]) # 생각해보니까 내 모델은 단일 오믹스라 omics_cls가 필요 없다.
        logit_mask = self.mom_head(fused[:, 1, :])

        return logit_cls, logit_mask

    def forward(self, x):
        img_cls, omics_cls, img_embed, omics_embed = self.forward_features(x)
        return img_cls, omics_cls, img_embed, omics_embed


def vit_base_patch16(rna_dim: int = 2000, **kwargs):
    model = VisionTransformer(
        rna_dim = rna_dim,
        img_size = 256,
        patch_size = 16,
        embed_dim = 384,
        depth = 6,
        num_heads = 8,
        mlp_ratio = 4,
        qkv_bias = True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    return model

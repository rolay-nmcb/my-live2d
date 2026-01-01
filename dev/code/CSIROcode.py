import os
import gc
import math
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import timm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 忽略警告信息
warnings.filterwarnings('ignore')

# =============================================================================
# 配置 
# 定义文件路径、超参数和硬件设置
# =============================================================================
class CFG:
    BASE_PATH = "/kaggle/input/csiro-biomass"
    TEST_CSV = os.path.join(BASE_PATH, "test.csv")
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, "test")
    # 模型权重所在的目录
    EXPERIMENT_DIR = "/kaggle/input/csiro/pytorch/default/12"
    
    N_FOLDS = 5
    # 优化1: 增大 Batch Size (如果显存不足报 OOM，请改成 4)
    BATCH_SIZE = 8 
    # 保持 0 以避免多进程报错，依靠 Batch Size 提升速度
    NUM_WORKERS = 0 
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 优化2: 关闭 TTA (测试时增强)，关闭后可获得 3x 提速，但精度可能微降
    USE_TTA = False 
    
    # 需要预测的目标列
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

# 设置日志记录器
LOGGER = logging.getLogger("CSIRO_Infer")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    LOGGER.addHandler(handler)

# =============================================================================
# 模型定义
# 这里包含了混合架构的各个组件：Transformer, MobileViT, PVT, Mamba
# =============================================================================

@dataclass
class ModelConfig:
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    # 候选骨干网络列表 (DINOv2 系列)
    dino_candidates: Tuple[str, ...] = ("vit_base_patch14_dinov2", "vit_base_patch14_reg4_dinov2", "vit_small_patch14_dinov2")
    # 用于切分图像的网格大小 (小尺度和大尺度)
    small_grid: Tuple[int, int] = (4, 4)
    big_grid: Tuple[int, int] = (2, 2)
    t2t_depth: int = 2
    cross_layers: int = 2
    cross_heads: int = 6
    # 特征融合金字塔的维度
    pyramid_dims: Tuple[int, int, int] = (384, 512, 640)
    mobilevit_heads: int = 4
    mobilevit_depth: int = 2
    sra_heads: int = 8
    sra_ratio: int = 2
    mamba_depth: int = 3
    mamba_kernel: int = 5
    aux_head: bool = True

MODEL_CFG = ModelConfig()

# 标准的前馈神经网络 (MLP)
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

# 标准的 Transformer 注意力块
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)
    def forward(self, x):
        h = self.norm1(x)
        # Self-Attention
        x = x + self.attn(h, h, h, need_weights=False)[0]
        # Feed Forward
        x = x + self.ff(self.norm2(x))
        return x

# MobileViT 模块：结合了 CNN 的局部特征提取和 Transformer 的全局建模能力
class MobileViTBlock(nn.Module):
    def __init__(self, dim, heads=4, depth=2, patch=(2, 2), dropout=0.0):
        super().__init__()
        # 局部特征提取 (CNN)
        self.local = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.Conv2d(dim, dim, 1), nn.GELU())
        self.patch = patch
        # 全局特征处理 (Transformer)
        self.transformer = nn.ModuleList([AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)])
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        local_feat = self.local(x)
        B, C, H, W = local_feat.shape
        ph, pw = self.patch
        # 确保特征图尺寸能被 Patch 大小整除，否则进行插值调整
        new_h, new_w = math.ceil(H / ph) * ph, math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w
        # 展开成 Patch 并通过 Transformer
        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw).contiguous().view(B, C, -1, ph, pw).permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        for blk in self.transformer: tokens = blk(tokens)
        # 恢复回特征图形状
        feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2).view(B, C, H // ph, W // pw, ph, pw).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # 将原始特征与处理后的特征拼接融合
        return self.fuse(torch.cat([x, feat], dim=1))

# 空间缩减注意力机制 (Spatial Reduction Attention) - PVT 的核心
# 通过减少 Key 和 Value 的空间尺寸来降低计算复杂度
class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        # 空间缩减层
        self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio) if sr_ratio > 1 else None
        self.norm = nn.LayerNorm(dim) if sr_ratio > 1 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, hw):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        # 如果需要缩减，将 Token 还原回空间结构进行卷积
        if self.sr is not None:
            feat = x.transpose(1, 2).reshape(B, C, hw[0], hw[1])
            feat = self.norm(self.sr(feat).reshape(B, C, -1).transpose(1, 2))
        else: feat = x
        k, v = self.kv(feat).chunk(2, dim=-1)
        k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attn = (torch.matmul(q, k) * self.scale).softmax(dim=-1)
        return self.proj(torch.matmul(self.drop(attn), v).permute(0, 2, 1, 3).reshape(B, N, C))

# PVT 块：包含 LayerNorm, SRA 注意力和 FeedForward
class PVTBlock(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sra = SpatialReductionAttention(dim, heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)
    def forward(self, x, hw):
        return self.ff(self.norm2(x + self.sra(self.norm1(x), hw))) + (x + self.sra(self.norm1(x), hw))

# 局部 Mamba 块：基于状态空间模型 (SSM) 的轻量级序列建模
class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 使用 1D 深度卷积模拟局部依赖
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.gate, self.proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        # 门控机制 + 卷积
        x = (self.dwconv((x * torch.sigmoid(self.gate(x))).transpose(1, 2)).transpose(1, 2))
        return shortcut + self.drop(self.proj(x))

# T2T (Token-to-Token) 模块：通过 Transformer 层重新聚合特征
class T2TRetokenizer(nn.Module):
    def __init__(self, dim, depth=2, heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock(dim, heads, dropout, 2.0) for _ in range(depth)])
    def forward(self, tokens, grid_hw):
        B, T, C = tokens.shape
        # 将 Token 还原为空间结构
        seq = tokens.transpose(1, 2).reshape(B, C, grid_hw[0], grid_hw[1]).flatten(2).transpose(1, 2)
        for blk in self.blocks: seq = blk(seq)
        # 通过自适应池化进行降采样
        return F.adaptive_avg_pool2d(seq.transpose(1, 2).reshape(B, C, grid_hw[0], grid_hw[1]), (2, 2)).flatten(2).transpose(1, 2), None

# 跨尺度融合模块：融合小尺度(grid)和大尺度(grid)的特征
class CrossScaleFusion(nn.Module):
    def __init__(self, dim, heads=6, dropout=0.0, layers=2):
        super().__init__()
        self.layers_s = nn.ModuleList([AttentionBlock(dim, heads, dropout, 2.0) for _ in range(layers)])
        self.layers_b = nn.ModuleList([AttentionBlock(dim, heads, dropout, 2.0) for _ in range(layers)])
        # 交叉注意力 (Cross Attention)
        self.cross_s = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim) for _ in range(layers)])
        self.cross_b = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim) for _ in range(layers)])
        self.norm_s, self.norm_b = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def forward(self, tok_s, tok_b):
        B, C = tok_s.shape[0], tok_s.shape[2]
        # 添加 CLS Token (Class Token)
        tok_s = torch.cat([tok_s.new_zeros(B, 1, C), tok_s], dim=1)
        tok_b = torch.cat([tok_b.new_zeros(B, 1, C), tok_b], dim=1)
        for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
            tok_s, tok_b = ls(tok_s), lb(tok_b)
            qs, qb = self.norm_s(tok_s[:, :1]), self.norm_b(tok_b[:, :1])
            # 交换信息：S 查询 B，B 查询 S
            tok_s = torch.cat([tok_s[:, :1] + cs(qs, torch.cat([tok_b, qb], 1), torch.cat([tok_b, qb], 1), need_weights=False)[0], tok_s[:, 1:]], 1)
            tok_b = torch.cat([tok_b[:, :1] + cb(qb, torch.cat([tok_s, qs], 1), torch.cat([tok_s, qs], 1), need_weights=False)[0], tok_b[:, 1:]], 1)
        # 移除 CLS Token 后合并返回
        return torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)

# Tile Encoder: 将大图切分成小块 (Tiles)，分别通过 Backbone 处理
class TileEncoder(nn.Module):
    def __init__(self, backbone, input_res):
        super().__init__()
        self.backbone, self.input_res = backbone, input_res
    def forward(self, x, grid):
        B, C, H, W = x.shape
        r, c = grid
        # 计算切分坐标
        hs, ws = torch.linspace(0, H, r + 1, device=x.device).long(), torch.linspace(0, W, c + 1, device=x.device).long()
        tiles = []
        for i in range(r):
            for j in range(c):
                xt = x[:, :, hs[i]:hs[i+1], ws[j]:ws[j+1]]
                # 调整 Tile 大小以适配 Backbone 输入
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, (self.input_res, self.input_res), mode="bilinear", align_corners=False)
                tiles.append(xt)
        # 批量通过 Backbone，然后重塑形状
        return self.backbone(torch.stack(tiles, 1).view(-1, C, self.input_res, self.input_res)).view(B, -1, self.backbone.num_features)

# 金字塔混合器：级联 MobileViT -> PVT -> Mamba
class PyramidMixer(nn.Module):
    def __init__(self, dim_in, dims, mobilevit_heads, mobilevit_depth, sra_heads, sra_ratio, mamba_depth, mamba_kernel, dropout):
        super().__init__()
        c1, c2, c3 = dims
        self.proj1 = nn.Linear(dim_in, c1)
        self.mobilevit = MobileViTBlock(c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout)
        self.proj2 = nn.Linear(c1, c2)
        self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
        self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)
        self.proj3 = nn.Linear(c2, c3)
        self.mamba_global = nn.ModuleList([LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)])
        self.final_attn = AttentionBlock(c3, heads=min(8, c3//64+1), dropout=dropout, mlp_ratio=2.0)

    # 关键函数：负责补齐 Token，防止在 reshape 为网格时因为数量不匹配报错
    def _pad_tokens(self, tokens, target_hw):
        B, N, C = tokens.shape
        H, W = target_hw
        need = H * W
        if N < need:
            pad = tokens.new_zeros(B, need - N, C)
            tokens = torch.cat([tokens, pad], dim=1)
        return tokens

    def forward(self, tokens):
        B, N, C = tokens.shape
        
        # --- Stage 1: MobileViT 处理 ---
        map_hw = (3, 4)
        t1 = self.proj1(tokens)
        t1_padded = self._pad_tokens(t1, map_hw) # 执行 Padding
        t1_map = t1_padded.transpose(1, 2).reshape(B, -1, map_hw[0], map_hw[1])
        
        m1 = self.mobilevit(t1_map)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :N]
        
        # --- Stage 2: PVT 处理 ---
        t2 = self.proj2(t1_out)
        t2 = t2[:, :N//2] + F.adaptive_avg_pool1d(t2.transpose(1, 2), N//2).transpose(1, 2)
        
        # 动态计算 Grid 大小并进行 Padding
        curr_n = t2.shape[1]
        h = int(math.sqrt(curr_n))
        w = h
        while h * w < curr_n:
            w += 1
            if h * w < curr_n: h += 1
        
        t2 = self._pad_tokens(t2, (h, w)) # 再次执行 Padding
        t2 = self.pvt(t2, (h, w))
        t2 = self.mamba_local(t2)

        # --- Stage 3: Global Mamba 处理 ---
        t3 = self.proj3(t2)
        # 结合均值池化和最大池化特征
        t3 = torch.stack([t3.mean(1), t3.max(1).values], 1)
        for blk in self.mamba_global: t3 = blk(t3)
        return self.final_attn(t3).mean(1), {"stage2_tokens": t2}

# 主模型类：整合 DINOv2 骨干网络、T2T、TileEncoder 和 PyramidMixer
class CrossPVT_T2T_MambaDINO(nn.Module):
    def __init__(self, dropout=0.1, hidden_ratio=0.35):
        super().__init__()
        # 构建 DINO 骨干网络
        self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino()
        # 图像切片编码器
        self.tile_encoder = TileEncoder(self.backbone, self.input_res)
        # Token 重组
        self.t2t = T2TRetokenizer(self.feat_dim, MODEL_CFG.t2t_depth, MODEL_CFG.cross_heads, dropout)
        # 跨尺度融合
        self.cross = CrossScaleFusion(self.feat_dim, MODEL_CFG.cross_heads, dropout, MODEL_CFG.cross_layers)
        
        # 金字塔混合头
        self.pyramid = PyramidMixer(self.feat_dim, MODEL_CFG.pyramid_dims, MODEL_CFG.mobilevit_heads, MODEL_CFG.mobilevit_depth,
                                    MODEL_CFG.sra_heads, MODEL_CFG.sra_ratio, MODEL_CFG.mamba_depth, MODEL_CFG.mamba_kernel, dropout)
        
        combined = MODEL_CFG.pyramid_dims[-1] * 2
        hidden = max(32, int(combined * hidden_ratio))
        
        # 定义输出头 (Green, Clover, Dead)
        def head(): return nn.Sequential(nn.Linear(combined, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.score_head = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, 1))
        # 左右视角融合的门控参数
        self.cross_gate_left = nn.Linear(MODEL_CFG.pyramid_dims[-1], MODEL_CFG.pyramid_dims[-1])
        self.cross_gate_right = nn.Linear(MODEL_CFG.pyramid_dims[-1], MODEL_CFG.pyramid_dims[-1])
        self.aux_head = nn.Sequential(nn.LayerNorm(MODEL_CFG.pyramid_dims[1]), nn.Linear(MODEL_CFG.pyramid_dims[1], 5)) if MODEL_CFG.aux_head else None
        # Softplus 激活函数保证输出为正数
        self.softplus = nn.Softplus()

    # 初始化 DINO 模型
    def _build_dino(self):
        for name in MODEL_CFG.dino_candidates:
            try:
                m = timm.create_model(name, pretrained=False, num_classes=0)
                if hasattr(m, "set_grad_checkpointing"): m.set_grad_checkpointing(True)
                input_res = 518
                if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
                    img_size = m.patch_embed.img_size
                    input_res = img_size[0] if isinstance(img_size, tuple) else img_size
                return m, m.num_features, name, input_res
            except Exception: continue
        raise RuntimeError("Backbone creation failed")

    # 单路特征提取流程
    def _half_forward(self, x):
        t_s, t_b = self.tile_encoder(x, MODEL_CFG.small_grid), self.tile_encoder(x, MODEL_CFG.big_grid)
        t2, _ = self.t2t(t_s, MODEL_CFG.small_grid)
        feat, maps = self.pyramid(self.cross(t2, t_b))
        return feat, maps

    # 前向传播
    def forward(self, x_cat, return_features=False):
        # 将输入拼接图像 (2000宽) 切分为左右两半 (1000x1000)
        W = x_cat.shape[-1] // 2
        x_left, x_right = x_cat[..., :W], x_cat[..., W:]
        
        # 分别处理左右图像
        fl, map_l = self._half_forward(x_left)
        fr, map_r = self._half_forward(x_right)
        
        # 使用门控机制融合左右特征
        fl = fl * torch.sigmoid(self.cross_gate_left(fr))
        fr = fr * torch.sigmoid(self.cross_gate_right(fl))
        f = torch.cat([fl, fr], dim=1)
        
        # 输出预测
        green = self.softplus(self.head_green(f))
        clover = self.softplus(self.head_clover(f))
        dead = self.softplus(self.head_dead(f))
        
        # 根据业务逻辑计算总生物量
        gdm = green + clover
        total = gdm + dead
        return {"total": total, "gdm": gdm, "green": green}

# =============================================================================
# 数据加载与推理部分
# =============================================================================

class TestBiomassDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]
        img = cv2.imread(os.path.join(self.image_dir, os.path.basename(path)))
        # 如果读不到图片，创建全黑图片防止崩溃
        img = np.zeros((1000, 2000, 3), dtype=np.uint8) if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(image=img)["image"]
        return img, path

# 定义图像增强/预处理：Resize 和 标准化
def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size * 2, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 加载保存的模型权重
def load_models():
    models = []
    input_res = 518
    LOGGER.info(f"Loading models from: {CFG.EXPERIMENT_DIR}")
    for fold in range(CFG.N_FOLDS):
        # 尝试不同路径格式寻找 checkpoint
        paths = [
            os.path.join(CFG.EXPERIMENT_DIR, f"fold_{fold}", "checkpoints", "best_wr2.pt"),
            os.path.join(CFG.EXPERIMENT_DIR, f"fold{fold}", "checkpoints", "best_wr2.pt")
        ]
        ckpt_path = next((p for p in paths if os.path.exists(p)), None)
        if not ckpt_path: continue
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            # 恢复配置
            if "cfg" in state:
                d = state["cfg"]
                MODEL_CFG.dropout = d.get("dropout", 0.1)
                MODEL_CFG.hidden_ratio = d.get("hidden_ratio", 0.35)
                if "pyramid_dims" in d: MODEL_CFG.pyramid_dims = d["pyramid_dims"]
            
            model = CrossPVT_T2T_MambaDINO(dropout=MODEL_CFG.dropout, hidden_ratio=MODEL_CFG.hidden_ratio)
            state_dict = state.get("model_state", state)
            # 移除 'module.' 前缀 (如果是 DataParallel 保存的)
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            model.to(CFG.DEVICE).eval()
            models.append(model)
            input_res = model.input_res
            LOGGER.info(f"Loaded Fold {fold} | Res: {input_res} | Dims: {MODEL_CFG.pyramid_dims}")
        except Exception as e: LOGGER.error(f"Failed to load fold {fold}: {e}")
    
    if not models: raise RuntimeError("No models loaded!")
    CFG.IMAGE_SIZE = input_res
    return models

# 推理函数
@torch.inference_mode()
def inference_fn(models, loader):
    results = []
    # 优化3: 启用混合精度推理 (自动选择 float16/float32)
    use_amp = torch.cuda.is_available()
    
    for imgs, paths in tqdm(loader, desc="Inferencing"):
        imgs = imgs.to(CFG.DEVICE, non_blocking=True)
        batch_size = imgs.shape[0]
        
        # 处理 TTA (Test Time Augmentation)：如果启用，生成翻转后的图片
        if CFG.USE_TTA:
            imgs_tta = torch.cat([imgs, torch.flip(imgs, dims=[3]), torch.flip(imgs, dims=[2])], dim=0)
        else:
            imgs_tta = imgs
            
        accum_preds = torch.zeros((imgs_tta.shape[0], 5), device=CFG.DEVICE)
        
        # 混合精度上下文，节省显存并提速
        with torch.amp.autocast('cuda', enabled=use_amp):
            for model in models:
                out = model(imgs_tta)
                gdm, green, total = out["gdm"], out["green"], out["total"]
                # 根据逻辑关系推导其他变量
                clover, dead = gdm - green, total - gdm
                preds = torch.cat([green, dead, clover, gdm, total], dim=1)
                # 累加预测结果，并确保非负
                accum_preds += torch.clamp(preds, min=0)
        
        # 取所有模型(Folds)的平均值
        accum_preds /= len(models)
        
        # 如果启用了 TTA，需要将翻转的预测结果平均回去
        if CFG.USE_TTA:
            accum_preds = accum_preds.view(3, batch_size, 5).mean(dim=0)
            
        accum_preds = accum_preds.float().cpu().numpy()
        
        # 保存结果
        for i, path in enumerate(paths):
            row = accum_preds[i]
            results.append({
                "image_path": path,
                "Dry_Green_g": row[0], "Dry_Dead_g": row[1], "Dry_Clover_g": row[2],
                "GDM_g": row[3], "Dry_Total_g": row[4]
            })
    return results

# =============================================================================
# 主函数+文件提交
# =============================================================================

def main():
    if not os.path.exists(CFG.TEST_CSV): raise FileNotFoundError("Test CSV not found")
    # 读取数据
    df = pd.read_csv(CFG.TEST_CSV)
    df_unique = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    # 加载模型
    models = load_models()
    dataset = TestBiomassDataset(df_unique, CFG.TEST_IMAGE_DIR, get_transforms(CFG.IMAGE_SIZE))
    # 创建 DataLoader
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    # 执行推理
    predictions = inference_fn(models, loader)
    # 格式化输出为 Kaggle 提交格式
    pred_df = pd.DataFrame(predictions)
    pred_melt = pred_df.melt(id_vars=["image_path"], value_vars=CFG.ALL_TARGET_COLS, var_name="target_name", value_name="target")
    submission = pd.merge(df[["sample_id", "image_path", "target_name"]], pred_melt, on=["image_path", "target_name"], how="left")
    submission["target"] = submission["target"].fillna(0.0)
    submission[["sample_id", "target"]].to_csv("submission.csv", index=False)
    LOGGER.info("Inference Complete!")
    LOGGER.info(submission.head())

if __name__ == "__main__":
    try: main()
    except Exception as e:
        # 异常处理：如果推理失败，生成全 0 的提交文件以避免评分错误
        LOGGER.error(f"Fatal Error: {e}")
        try: pd.read_csv(CFG.TEST_CSV)[["sample_id"]].assign(target=0.0).to_csv("submission.csv", index=False)
        except: pass
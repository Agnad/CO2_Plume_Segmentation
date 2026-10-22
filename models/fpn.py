import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .denoiser import ConvTransformerBlock


class FeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid Network (FPN) with Channel-Wise Recalibration Gating.
    
    Reuses hierarchical representations F1, F2, F3 from the shared ConvTransformer encoder:
    1. Lateral Convolutions: 1x1 Conv projects encoder channels [32, 64, 128] to unified 256 channels.
    2. Top-Down Fusion: Bilinear 2x upsampling and element-wise addition propagate high-level context.
    3. Channel Recalibration: Adaptive global average pooling + 1x1 Conv + Sigmoid channel gating.
    4. Multi-Scale Fusion: 3x3 Conv + GroupNorm(8) + GELU reduces each level from 256 to 128 channels.
    5. Spatial Alignment & Concatenation: Lower resolution maps are upsampled to the resolution of F1
       and concatenated along channels to form a 384-channel fused representation.
       
    Args:
        enc_dims (List[int]): Input channel counts of the 3 encoder stages (default: [32, 64, 128]).
        mid_ch (int): Unified lateral channel dimension (default: 256).
    """
    def __init__(self, enc_dims: List[int] = [32, 64, 128], mid_ch: int = 256):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(d, mid_ch, 1) for d in enc_dims])
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch // 2, 3, padding=1),  # 256 -> 128
                nn.GroupNorm(8, mid_ch // 2),
                nn.GELU()
            ) for _ in enc_dims
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_ch, mid_ch, 1),
                nn.Sigmoid()
            ) for _ in enc_dims
        ])

    def forward(self, feats: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Forward pass for multi-scale feature pyramid fusion.
        
        Args:
            feats: Sequence of encoder tensors [F1, F2, F3]
                   F1: (B, 32, H, W)
                   F2: (B, 64, H/2, W/2)
                   F3: (B, 128, H/4, W/4)
                   
        Returns:
            Tensor: Concatenated multi-scale feature map (B, 384, H, W).
        """
        # Unwrap nested tuple/list if passed through checkpoint lambda
        if isinstance(feats, (list, tuple)) and len(feats) == 1 and isinstance(feats[0], (list, tuple)):
            feats = feats[0]

        # Lateral projections
        p3 = self.laterals[0](feats[0])  # (B, 256, H, W)
        p4 = self.laterals[1](feats[1])  # (B, 256, H/2, W/2)
        p5 = self.laterals[2](feats[2])  # (B, 256, H/4, W/4)

        # Top-down recursive fusion
        p4 = p4 + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = p3 + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)

        # Channel-wise recalibration gating
        g3 = self.gates[0](p3)
        g4 = self.gates[1](p4)
        g5 = self.gates[2](p5)

        # Fusion convolution and spatial alignment to F1 resolution
        f1_out = self.fusions[0](p3 * g3)
        f2_out = self.fusions[1](F.interpolate(p4 * g4, size=p3.shape[2:], mode='bilinear', align_corners=False))
        f3_out = self.fusions[2](F.interpolate(p5 * g5, size=p3.shape[2:], mode='bilinear', align_corners=False))

        # Concatenate along channel dimension: 128 + 128 + 128 = 384 channels
        return torch.cat([f1_out, f2_out, f3_out], dim=1)


class SegmentationHead(nn.Module):
    """
    Plume Segmentation Head.
    
    Processes concatenated 384-channel multi-scale representations through:
    1. ConvTransformerBlock (384 ch, 12 heads, window 8)
    2. Grouped Conv2d (384 ch, 128 groups, kernel 3x3) + GELU
    3. Gradual channel reduction: 384 -> 256 -> 128 -> 1
    4. Sigmoid activation producing pixel-wise plume probability map in [0, 1].
    """
    def __init__(self, in_channels: int = 384):
        super().__init__()
        self.block = ConvTransformerBlock(in_channels, num_heads=12, window_size=8)
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=128),
            nn.GELU()
        )
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for plume segmentation prediction.
        
        Args:
            x (Tensor): Multi-scale concatenated feature tensor (B, 384, H, W).
            
        Returns:
            Tensor: Calibrated plume probability map (B, 1, H, W) in [0, 1].
        """
        x = self.block(x)
        x = self.grouped_conv(x)
        return self.reduction(x)

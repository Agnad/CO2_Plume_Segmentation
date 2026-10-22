import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Optional


class WindowAttention(nn.Module):
    """
    Local Window-based Multi-Head Self-Attention with Learnable Relative Position Bias.
    
    Operates within non-overlapping or cyclically shifted local spatial windows of size
    (window_size x window_size), computing scaled dot-product attention while maintaining
    linear computational complexity with respect to image resolution.
    
    Args:
        dim (int): Number of feature channels.
        num_heads (int): Number of attention heads.
        window_size (int): Spatial dimension of the local square window.
        shift_size (int): Number of pixels to cyclically shift for cross-window connectivity.
    """
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 16, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Learnable relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords = coords.flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            'relative_position_index',
            relative_coords.sum(-1).flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for window-based multi-head self-attention.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            Tensor: Attended feature tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        # Pad feature map to integer multiples of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_pad = F.pad(shifted_x, (0, 0, 0, pad_r, 0, pad_b))
        
        # Partition into windows
        x_windows = rearrange(
            x_pad, "b (h wh) (w ww) c -> (b h w) (wh ww) c",
            wh=self.window_size, ww=self.window_size
        )

        # Project Q, K, V
        qkv = self.qkv(x_windows).view(
            -1, self.window_size ** 2, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        bias = self.relative_position_bias_table[self.relative_position_index]
        bias = bias.view(1, self.num_heads, self.window_size ** 2, self.window_size ** 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size ** 2, C)
        
        # Merge windows back
        x = rearrange(
            x, "(b h w) (wh ww) c -> b (h wh) (w ww) c",
            h=x_pad.shape[1] // self.window_size,
            w=x_pad.shape[2] // self.window_size,
            wh=self.window_size, ww=self.window_size
        )
        
        # Unpad and unshift
        x = x[:, :H, :W, :].contiguous()
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
            
        return self.proj(x).permute(0, 3, 1, 2)


class ConvTransformerBlock(nn.Module):
    """
    Hierarchical ConvTransformer Block.
    
    Comprises two sub-stages with residual skip connections:
    1. Group Normalization (1 group) -> Local Window Multi-Head Self-Attention
    2. Group Normalization (1 group) -> Two-layer Pointwise MLP (4x expansion) with GELU
    
    Args:
        dim (int): Number of input/output feature channels.
        num_heads (int): Number of attention heads.
        window_size (int): Size of local attention window.
        shift_size (int): Cyclic shift offset (0 for standard window, window_size // 2 for shifted).
    """
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = WindowAttention(dim, num_heads, window_size, shift_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.use_checkpoint = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientDenoiser(nn.Module):
    """
    Hierarchical ConvTransformer Denoising Backbone.
    
    Encoder-decoder network for spaceborne XCO2 concentration restoration:
    - Stem: 7x7 Conv (1 -> 32 channels) + GroupNorm(8) + GELU
    - Downsampling Stage 1: 3x3 Stride-2 Conv (32 -> 64) + 2x ConvTransformer blocks
    - Downsampling Stage 2: 3x3 Stride-2 Conv (64 -> 128) + 2x ConvTransformer blocks
    - Bottleneck: 6x ConvTransformer blocks with alternating shifted windows
    - Upsampling Stage 1: 2x Nearest-neighbor + 3x3 Conv (128 -> 64) + 2x blocks + Skip from Down1
    - Upsampling Stage 2: 2x Nearest-neighbor + 3x3 Conv (64 -> 32) + 2x blocks + Skip from Stem
    - Continuous Reconstruction Head: 3x3 Conv (32 -> 1 channel) without activation
    """
    def __init__(
        self,
        in_ch: int = 1,
        dims: List[int] = [32, 64, 128],
        depths: List[int] = [2, 2, 6],
        window_sizes: List[int] = [8, 4, 4, 8, 16],
        num_heads: List[int] = [4, 8, 8, 4, 2]
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 7, padding=3),
            nn.GroupNorm(8, dims[0]),
            nn.GELU()
        )
        
        self.down1 = self._make_stage(dims[0], dims[1], depths[0], window_sizes[0], num_heads[0])
        self.down2 = self._make_stage(dims[1], dims[2], depths[1], window_sizes[1], num_heads[1])
        
        self.bottleneck = nn.Sequential(*[
            ConvTransformerBlock(
                dims[2], num_heads[2], window_sizes[2],
                shift_size=(window_sizes[2] // 2 if i % 2 == 0 else 0)
            ) for i in range(depths[2])
        ])
        
        self.up1 = self._make_stage(dims[2], dims[1], depths[1], window_sizes[3], num_heads[3], upsample=True)
        self.up2 = self._make_stage(dims[1], dims[0], depths[0], window_sizes[4], num_heads[4], upsample=True)
        
        self.head = nn.Conv2d(dims[0], 1, 3, padding=1)

    def _make_stage(
        self, in_d: int, out_d: int, depth: int, ws: int, nh: int, upsample: bool = False
    ) -> nn.Sequential:
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(in_d, out_d, 3, padding=1))
        else:
            layers.append(nn.Conv2d(in_d, out_d, 3, 2, 1))
            
        layers += [ConvTransformerBlock(out_d, nh, ws) for _ in range(depth)]
        return nn.Sequential(*layers)

    def enable_gradient_checkpointing(self):
        """Enables gradient checkpointing in all constituent ConvTransformer blocks."""
        for m in self.modules():
            if isinstance(m, ConvTransformerBlock):
                m.use_checkpoint = True

    def disable_gradient_checkpointing(self):
        """Disables gradient checkpointing in all constituent ConvTransformer blocks."""
        for m in self.modules():
            if isinstance(m, ConvTransformerBlock):
                m.use_checkpoint = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standalone forward pass of the denoiser.
        
        Args:
            x (Tensor): Single-channel noisy XCO2 observation (B, 1, H, W).
            
        Returns:
            Tensor: Continuous restored XCO2 field (B, 1, H, W).
        """
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xb = self.bottleneck(x2)
        x = self.up1(xb) + x1
        x = self.up2(x) + x0
        return self.head(x)

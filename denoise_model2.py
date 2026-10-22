# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class WindowAttention(nn.Module):
    """Memory-efficient windowed attention with shifted windows"""
    def __init__(self, dim, num_heads=8, window_size=16, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("relative_position_bias", self.create_bias_table())

    def create_bias_table(self):
        window_size = self.window_size
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        )).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_index = relative_coords.sum(-1)
        
        max_rel_dist = (2 * window_size - 1) ** 2
        bias_table = torch.zeros(max_rel_dist, self.num_heads)
        nn.init.trunc_normal_(bias_table, std=0.02)
        return bias_table[relative_index.view(-1)].view(
            window_size**2, window_size**2, -1).permute(2, 0, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_pad = F.pad(shifted_x, (0, 0, 0, pad_r, 0, pad_b))
        
        x_windows = rearrange(x_pad, "b (h wh) (w ww) c -> b (h w) (wh ww) c",
                            wh=self.window_size, ww=self.window_size)
        x_windows = x_windows.view(-1, self.window_size**2, C)

        qkv = self.qkv(x_windows).view(
            -1, self.window_size**2, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.relative_position_bias
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size**2, C)
        
        x = rearrange(x, "(b h w) (wh ww) c -> b (h wh) (w ww) c",
                    h=x_pad.shape[1]//self.window_size,
                    w=x_pad.shape[2]//self.window_size,
                    wh=self.window_size, ww=self.window_size)
        
        x = x[:, :H, :W, :].contiguous()
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
            
        return self.proj(x).permute(0, 3, 1, 2)

class ConvTransformerBlock(nn.Module):
    """Hybrid Conv-Transformer Block with Checkpointing"""
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = WindowAttention(dim, num_heads, window_size, shift_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )
        self.use_checkpoint = False

    def forward(self, x):
        if self.use_checkpoint:
            return self._checkpoint_forward(x)
        return self._original_forward(x)

    def _original_forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _checkpoint_forward(self, x):
        x = x + checkpoint(self.attn, self.norm1(x), use_reentrant=False)
        x = x + checkpoint(self.mlp, self.norm2(x), use_reentrant=False)
        return x

class EfficientDenoiser(nn.Module):
    """Memory-efficient hybrid transformer-conv denoising model"""
    def __init__(self, in_chans=1, dims=[32, 64, 128], depths=[2, 2, 6]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 7, padding=3),
            nn.GroupNorm(8, dims[0]),
            nn.GELU()
        )
        
        # Downsample
        self.down1 = self._make_down_layer(dims[0], dims[1], depths[0], window=8, shift_cycle=2)
        self.down2 = self._make_down_layer(dims[1], dims[2], depths[1], window=4, shift_cycle=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            ConvTransformerBlock(dims[2], 8, 4, 0 if i%2==0 else 2)
            for i in range(depths[2])
        ])
        
        # Upsample
        self.up1 = self._make_up_layer(dims[2], dims[1], depths[1], window=8)
        self.up2 = self._make_up_layer(dims[1], dims[0], depths[0], window=16)
        
        self.head = nn.Conv2d(dims[0], 1, 3, padding=1)

    def _make_down_layer(self, in_dim, out_dim, depth, window, shift_cycle):
        layers = [nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)]
        layers += [
            ConvTransformerBlock(out_dim, out_dim//16, window, 
                                shift_size=(0 if i%shift_cycle==0 else window//2))
            for i in range(depth)
        ]
        return nn.Sequential(*layers)

    def _make_up_layer(self, in_dim, out_dim, depth, window):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            *[ConvTransformerBlock(out_dim, out_dim//16, window)
              for _ in range(depth)]
        )

    def enable_gradient_checkpointing(self):
        for m in self.modules():
            if isinstance(m, ConvTransformerBlock):
                m.use_checkpoint = True

    def disable_gradient_checkpointing(self):
        for m in self.modules():
            if isinstance(m, ConvTransformerBlock):
                m.use_checkpoint = False
    
   

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = self.bottleneck(x2)
        x = self.up1(x) + x1
        x = self.up2(x) + x0
        return self.head(x)

class PhysicsInformedLoss(nn.Module):
    def __init__(self, tv_weight=0.1, mass_weight=0.2):
        super().__init__()
        self.tv_weight = tv_weight
        self.mass_weight = mass_weight

    def forward(self, pred, target, plume_mask=None):
        mse = F.mse_loss(pred, target)
        tv_loss = torch.mean(torch.abs(pred[..., 1:] - pred[..., :-1])) * 2
        mass_loss = 0.0
        
        if plume_mask is not None:
            mass_diff = (pred*plume_mask).sum((1,2,3)) - (target*plume_mask).sum((1,2,3))
            mass_loss = torch.mean(torch.abs(mass_diff))
            
        return mse + self.tv_weight*tv_loss + self.mass_weight*mass_loss
if __name__ == "__main__":
    from data_build_seed import get_loaders
    
    # Test configuration
    model = EfficientDenoiser()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(2, 1, 160, 160)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")


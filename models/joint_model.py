import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
from .denoiser import EfficientDenoiser
from .fpn import FeaturePyramid, SegmentationHead


class JointDenoiseSeg(nn.Module):
    """
    Restoration-Aware Joint Learning Framework for XCO2 Plume Segmentation.
    
    Couples XCO2 concentration restoration and plume footprint segmentation
    through a shared hierarchical ConvTransformer encoder under restricted
    backbone adaptation.
    
    Architecture Pathways:
    1. Shared Encoder: Extracts multi-scale representations (F1, F2, F3).
    2. Denoising Decoder: Reconstructs continuous restored XCO2 field via residual upsampling.
    3. Segmentation Pathway: Fuses multi-scale encoder representations through FPN and channel-wise
       recalibration gating, then predicts pixel-wise plume probability map.
       
    Restricted Backbone Adaptation:
    - Pretrained denoiser core convolutional kernels and attention projection weights are frozen.
    - Normalization layers (GroupNorm) and bias vectors remain trainable to adapt distributions.
    - Feature pyramid and segmentation head modules are trained end-to-end.
    
    Args:
        weights_path (str, optional): Path to pretrained EfficientDenoiser checkpoint.
        freeze_backbone (bool): Whether to enforce restricted backbone adaptation (default: True).
    """
    def __init__(self, weights_path: Optional[str] = None, freeze_backbone: bool = True):
        super().__init__()
        self.denoiser = EfficientDenoiser()
        
        # Load pretrained denoiser weights if provided
        if weights_path is not None:
            if os.path.isfile(weights_path):
                self._load_weights(weights_path)
            else:
                print(f"[Warning] Checkpoint '{weights_path}' not found. Initializing with random weights.")
                
        # Segmentation branch reusing hierarchical encoder representations
        self.fpn = FeaturePyramid(enc_dims=[
            self.denoiser.stem[0].out_channels,     # 32
            self.denoiser.down1[0].out_channels,    # 64
            self.denoiser.down2[0].out_channels     # 128
        ], mid_ch=256)
        
        self.seg_head = SegmentationHead(in_channels=384)
        
        if freeze_backbone:
            self._configure_gradients()

    def _load_weights(self, path: str):
        """Loads pretrained weights into the denoiser backbone with non-strict matching."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state_dict = state['model_state_dict']
        else:
            state_dict = state
            
        load_result = self.denoiser.load_state_dict(state_dict, strict=False)
        print(f"[JointDenoiseSeg] Loaded pretrained weights from '{path}'")
        print(f"  - Loaded keys: {len(state_dict) - len(load_result.missing_keys)} / {len(state_dict)}")
        print(f"  - Missing keys (expected for non-denoising modules): {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"  - Unexpected keys: {len(load_result.unexpected_keys)}")

    def _configure_gradients(self):
        """
        Enforces restricted backbone adaptation protocol:
        - Freezes encoder convolutional weights and attention linear layers.
        - Keeps normalization layers and bias parameters trainable.
        - Enables gradients on all FPN and segmentation head parameters.
        """
        for name, param in self.denoiser.named_parameters():
            if 'norm' in name or 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        for module in [self.fpn, self.seg_head]:
            for param in module.parameters():
                param.requires_grad = True

    def get_trainable_parameters_summary(self) -> dict:
        """Returns counts of frozen vs trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_pct': (trainable / total) * 100.0 if total > 0 else 0.0
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the joint framework.
        
        Args:
            x (Tensor): Single-channel noisy XCO2 observation tensor of shape (B, 1, H, W).
            
        Returns:
            Tuple[Tensor, Tensor]:
                - den (Tensor): Restored continuous XCO2 concentration field (B, 1, H, W).
                - seg (Tensor): Calibrated plume probability map in [0, 1] of shape (B, 1, H, W).
        """
        is_cuda = x.is_cuda
        
        # 1. Full-precision execution for encoder feature extraction and FPN
        if is_cuda:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                x_fp32 = x.float()
                x0 = checkpoint(self.denoiser.stem, x_fp32, use_reentrant=False)
                x1 = checkpoint(self.denoiser.down1, x0, use_reentrant=False)
                x2 = checkpoint(self.denoiser.down2, x1, use_reentrant=False)
                xb = checkpoint(self.denoiser.bottleneck, x2, use_reentrant=False)
                
                # FPN multi-scale feature fusion
                features = checkpoint(
                    lambda *args: self.fpn(args),
                    x0, x1, x2,
                    use_reentrant=False
                )
        else:
            x_fp32 = x.float()
            x0 = self.denoiser.stem(x_fp32)
            x1 = self.down1_block(x0) if hasattr(self, 'down1_block') else self.denoiser.down1(x0)
            x2 = self.denoiser.down2(x1)
            xb = self.denoiser.bottleneck(x2)
            features = self.fpn([x0, x1, x2])

        # 2. Decoder and segmentation head execution
        if is_cuda:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Continuous reconstruction decoder
                den = self.denoiser.head(
                    self.denoiser.up2(self.denoiser.up1(xb) + x1) + x0
                )
                # Multi-scale segmentation prediction
                seg = self.seg_head(features.to(torch.float16))
        else:
            den = self.denoiser.head(
                self.denoiser.up2(self.denoiser.up1(xb) + x1) + x0
            )
            seg = self.seg_head(features)

        return den, seg

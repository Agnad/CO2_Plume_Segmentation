"""
combi_seg_den.py - Top-level module exposing the Restoration-Aware Joint Model.

Maintains backward compatibility with legacy imports while forwarding to `models.joint_model`.
"""
import torch
from models.denoiser import WindowAttention, ConvTransformerBlock, EfficientDenoiser
from models.fpn import FeaturePyramid, SegmentationHead
from models.joint_model import JointDenoiseSeg

__all__ = [
    'WindowAttention',
    'ConvTransformerBlock',
    'EfficientDenoiser',
    'FeaturePyramid',
    'SegmentationHead',
    'JointDenoiseSeg',
]

if __name__ == '__main__':
    print("Testing JointDenoiseSeg architecture...")
    model = JointDenoiseSeg(weights_path=None, freeze_backbone=True)
    summary = model.get_trainable_parameters_summary()
    print(f"Total parameters:     {summary['total']:,}")
    print(f"Trainable parameters: {summary['trainable']:,} ({summary['trainable_pct']:.2f}%)")
    print(f"Frozen parameters:    {summary['frozen']:,}")
    
    dummy_input = torch.randn(2, 1, 160, 160)
    with torch.no_grad():
        den, seg = model(dummy_input)
    print(f"Denoised output shape: {den.shape}")
    print(f"Plume mask output shape: {seg.shape}")
    assert den.shape == (2, 1, 160, 160)
    assert seg.shape == (2, 1, 160, 160)
    print("Verification passed successfully.")

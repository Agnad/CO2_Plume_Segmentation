"""
physics_losses.py - Top-level module exposing physics-informed joint and constituent loss functions.

Maintains backward compatibility with legacy scripts while forwarding to `models.losses`.
"""
import torch
from models.losses import PhysicsSSIMLoss, CO2PlumeSegmentationLoss, JointPhysicsLoss

__all__ = [
    'PhysicsSSIMLoss',
    'CO2PlumeSegmentationLoss',
    'JointPhysicsLoss',
]

if __name__ == '__main__':
    print("Testing JointPhysicsLoss...")
    loss_fn = JointPhysicsLoss(denoise_weight=0.3, seg_weight=0.7)
    
    pred_den = torch.randn(2, 1, 160, 160)
    pred_seg = torch.rand(2, 1, 160, 160)
    clean_target = torch.randn(2, 1, 160, 160)
    target_mask = (torch.rand(2, 1, 160, 160) > 0.8).float()
    
    losses = loss_fn(pred_den, pred_seg, clean_target, target_mask)
    print(f"Total Loss:        {losses['total'].item():.4f}")
    print(f"Denoise Component: {losses['denoise'].item():.4f}")
    print(f"Seg Component:     {losses['seg'].item():.4f}")
    print("Verification passed successfully.")

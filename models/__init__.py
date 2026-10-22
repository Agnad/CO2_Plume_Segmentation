"""
Models package for Restoration-Aware Joint Denoising and Segmentation of Satellite XCO2 Plumes.
"""
from .denoiser import WindowAttention, ConvTransformerBlock, EfficientDenoiser
from .fpn import FeaturePyramid, SegmentationHead
from .joint_model import JointDenoiseSeg
from .losses import PhysicsSSIMLoss, CO2PlumeSegmentationLoss, JointPhysicsLoss

__all__ = [
    'WindowAttention',
    'ConvTransformerBlock',
    'EfficientDenoiser',
    'FeaturePyramid',
    'SegmentationHead',
    'JointDenoiseSeg',
    'PhysicsSSIMLoss',
    'CO2PlumeSegmentationLoss',
    'JointPhysicsLoss',
]

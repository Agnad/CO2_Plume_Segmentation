import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None


class PhysicsSSIMLoss(nn.Module):
    """
    Continuous XCO2 Reconstruction Loss with Physical Regularization.
    
    Combines:
    1. Pixel-wise Mean Squared Error (MSE)
    2. Structural Similarity Index Metric (SSIM)
    3. Continuous Total Variation Regularization (TV)
    4. Plume Footprint Mass Conservation (L_mass)
    
    Formula:
        L_den = 0.8 * MSE + 0.2 * (1 - SSIM) + 0.05 * L_TV + 0.1 * L_mass
        
    Args:
        tv_weight (float): Weight for total-variation regularization (default: 0.05).
        mass_weight (float): Weight for plume mass conservation (default: 0.1).
        ssim_weight (float): Weight for SSIM loss component (default: 0.2).
        data_range (float): Dynamic range of XCO2 concentrations in ppm (default: 6.0).
    """
    def __init__(
        self,
        tv_weight: float = 0.05,
        mass_weight: float = 0.1,
        ssim_weight: float = 0.2,
        data_range: float = 6.0
    ):
        super().__init__()
        self.tv_weight = tv_weight
        self.mass_weight = mass_weight
        self.ssim_weight = ssim_weight
        self.data_range = data_range

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes SSIM loss, using pytorch_msssim if available or Gaussian fallback."""
        if ssim is not None:
            return 1.0 - ssim(pred, target, data_range=self.data_range, size_average=True)
        # Fallback SSIM approximation if pytorch_msssim is not installed
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        mu_x = pred.mean(dim=[-2, -1], keepdim=True)
        mu_y = target.mean(dim=[-2, -1], keepdim=True)
        sigma_x = ((pred - mu_x) ** 2).mean(dim=[-2, -1], keepdim=True)
        sigma_y = ((target - mu_y) ** 2).mean(dim=[-2, -1], keepdim=True)
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=[-2, -1], keepdim=True)
        ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
        return 1.0 - ssim_val.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        plume_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates continuous denoising loss in float32.
        
        Args:
            pred (Tensor): Restored continuous concentration field (B, 1, H, W).
            target (Tensor): Clean ground-truth concentration field (B, 1, H, W).
            plume_mask (Tensor, optional): Binary or probabilistic plume footprint mask (B, 1, H, W).
            
        Returns:
            Tensor: Scalar loss value.
        """
        # Force FP32 execution for numerical stability
        pred = pred.float()
        target = target.float()
        
        # 1. Pixel-wise MSE
        mse = F.mse_loss(pred, target)
        
        # 2. Structural Similarity Loss
        ssim_loss = self._compute_ssim(pred, target)
        
        # 3. Total Variation Regularization
        tv_h = torch.abs(pred[..., 1:] - pred[..., :-1]).mean()
        tv_w = torch.abs(pred[..., 1:, :] - pred[..., :-1, :]).mean()
        tv_loss = tv_h + tv_w
        
        # 4. Plume Mass Conservation
        mass_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if plume_mask is not None:
            plume_mask = plume_mask.float()
            mass_pred = (pred * plume_mask).sum(dim=(1, 2, 3))
            mass_target = (target * plume_mask).sum(dim=(1, 2, 3))
            mass_loss = torch.abs(mass_pred - mass_target).mean()

        return (1.0 - self.ssim_weight) * mse + \
               self.ssim_weight * ssim_loss + \
               self.tv_weight * tv_loss + \
               self.mass_weight * mass_loss


class CO2PlumeSegmentationLoss(nn.Module):
    """
    Boundary-Aware Segmentation Loss with Atmospheric Physics Constraints.
    
    Combines:
    1. Boundary-Aware Focal Tversky Loss (L_FT) targeting ambiguous perimeter transitions.
    2. Atmospheric Background Physics Violation Penalty (L_phys) penalizing reconstructed
       concentrations below local baseline in predicted plume footprints.
    3. Mask Total Variation Regularization (L_TV-mask) suppressing isolated noise activations.
    
    Formula:
        L_seg = L_FT + 0.5 * L_phys + 0.1 * L_TV-mask
        
    Args:
        co2_thresh (float): Ambient regional background mole fraction threshold in ppm (default: 402.0).
        target_mean (float): Dataset mean parameter for physical denormalization (default: 402.0).
        target_std (float): Dataset standard deviation parameter for physical denormalization (default: 3.42).
        focal_alpha (float): False positive weight coefficient in Tversky index (default: 0.8).
        focal_gamma (float): Focal exponent focusing gradient on hard boundary errors (default: 2.0).
        boundary_weight (float): Multiplier for 5x5 morphological boundary band (default: 3.0).
        tv_weight (float): Weight for mask smoothness TV regularization (default: 0.1).
        phys_weight (float): Weight for atmospheric physics penalty (default: 0.5).
    """
    def __init__(
        self,
        co2_thresh: float = 402.0,
        target_mean: float = 402.0,
        target_std: float = 3.42,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        boundary_weight: float = 3.0,
        tv_weight: float = 0.1,
        phys_weight: float = 0.5
    ):
        super().__init__()
        self.register_buffer('co2_threshold', torch.tensor(co2_thresh, dtype=torch.float32))
        self.register_buffer('target_mean', torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer('target_std', torch.tensor(target_std, dtype=torch.float32))
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight
        self.tv_weight = tv_weight
        self.phys_weight = phys_weight

    def _compute_boundary_weights(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Extracts morphological dilation-erosion boundary band and creates spatial weights."""
        mask = mask.float()
        eroded = -F.max_pool2d(-mask, kernel_size, padding=kernel_size // 2, stride=1)
        dilated = F.max_pool2d(mask, kernel_size, padding=kernel_size // 2, stride=1)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        return boundary * self.boundary_weight + 1.0

    def _physics_constraint(self, denoised: torch.Tensor, pred_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes stabilized logarithmic-exponential physics violation penalty.
        Penalizes pixels where predicted plume mask is active (>0.5) but reconstructed
        XCO2 is below ambient background threshold.
        """
        # Convert denoised concentration to physical mole-fraction units (ppm)
        denorm = denoised.float() * self.target_std + self.target_mean
        plume_mask = (pred_prob > 0.5).float()
        
        # Positive violations where concentration falls below background
        violations = F.relu(self.co2_threshold - (denorm * plume_mask))
        
        # 1. Gradient-stabilizing value clamping at 15 ppm
        violations = torch.clamp(violations, max=15.0)
        
        # 2. Temperature-controlled exponential growth
        temperature = 8.0
        exp_term = torch.exp(violations / temperature)
        
        # 3. Logarithmic scaling for smooth asymptotic behavior
        log_scale = torch.log1p(violations)
        
        return torch.mean(log_scale * exp_term)

    def forward(
        self,
        denoised: torch.Tensor,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates plume segmentation loss.
        
        Args:
            denoised (Tensor): Restored continuous concentration field (B, 1, H, W).
            pred_mask (Tensor): Model segmentation output (B, 1, H, W).
            target_mask (Tensor): Ground-truth binary plume mask (B, 1, H, W).
            
        Returns:
            Tensor: Scalar loss value.
        """
        denoised = denoised.float()
        pred_mask = pred_mask.float()
        target_mask = target_mask.float()

        # Handle probabilities vs logits
        pred_prob = torch.sigmoid(pred_mask) if (pred_mask.min() < 0.0 or pred_mask.max() > 1.0) else pred_mask

        # 1. Boundary-weighted Focal Tversky Loss
        boundary_weights = self._compute_boundary_weights(target_mask)
        tp = (pred_prob * target_mask * boundary_weights).sum()
        fp = ((pred_prob * (1.0 - target_mask)) * boundary_weights).sum()
        fn = (((1.0 - pred_prob) * target_mask) * boundary_weights).sum()
        
        tversky = tp / (tp + self.focal_alpha * fp + (1.0 - self.focal_alpha) * fn + 1e-8)
        ft_loss = (1.0 - tversky) ** self.focal_gamma

        # 2. Atmospheric Physics Constraint
        phys_loss = self._physics_constraint(denoised, pred_prob)

        # 3. Mask Smoothness Total Variation Regularization
        tv_loss = torch.mean((pred_prob[..., 1:] - pred_prob[..., :-1]) ** 2) + \
                  torch.mean((pred_prob[..., 1:, :] - pred_prob[..., :-1, :]) ** 2)

        return ft_loss + self.phys_weight * phys_loss + self.tv_weight * tv_loss


class JointPhysicsLoss(nn.Module):
    """
    Joint Multi-Task Physics-Informed Optimization Criterion.
    
    Formula:
        L_total = lambda_den * L_den + lambda_seg * L_seg
        Default: 0.3 * L_den + 0.7 * L_seg
        
    Args:
        denoise_weight (float): Multiplier for continuous restoration loss (default: 0.3).
        seg_weight (float): Multiplier for plume segmentation loss (default: 0.7).
        denoise_kwargs (dict, optional): Custom keyword args for PhysicsSSIMLoss.
        seg_kwargs (dict, optional): Custom keyword args for CO2PlumeSegmentationLoss.
    """
    def __init__(
        self,
        denoise_weight: float = 0.3,
        seg_weight: float = 0.7,
        denoise_kwargs: Optional[dict] = None,
        seg_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.denoise_weight = denoise_weight
        self.seg_weight = seg_weight
        
        self.denoise_loss = PhysicsSSIMLoss(**(denoise_kwargs or {}))
        self.seg_loss = CO2PlumeSegmentationLoss(**(seg_kwargs or {}))

    def forward(
        self,
        denoised: torch.Tensor,
        seg_pred: torch.Tensor,
        clean: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-task joint loss and constituent sub-losses.
        
        Args:
            denoised (Tensor): Predicted restored XCO2 field (B, 1, H, W).
            seg_pred (Tensor): Predicted plume mask / probabilities (B, 1, H, W).
            clean (Tensor): Ground-truth clean XCO2 field (B, 1, H, W).
            mask (Tensor): Ground-truth binary plume footprint mask (B, 1, H, W).
            
        Returns:
            Dict[str, Tensor]: Dictionary containing:
                - 'total': Weighted combined scalar loss.
                - 'denoise': Denoising loss component.
                - 'seg': Segmentation loss component.
        """
        den_loss = self.denoise_loss(denoised, clean, mask)
        seg_loss = self.seg_loss(denoised, seg_pred, mask)
        
        return {
            'total': self.denoise_weight * den_loss + self.seg_weight * seg_loss,
            'denoise': den_loss.float(),
            'seg': seg_loss.float()
        }

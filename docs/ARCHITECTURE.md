# Deep Architecture & Mathematical Formulation

This document provides a comprehensive technical overview of the **Restoration-Aware Joint Denoising and Segmentation Framework** for satellite $\text{XCO}_2$ plume monitoring.

---

## 1. Network Architecture Overview

The system couples continuous $\text{XCO}_2$ concentration restoration and discrete plume footprint segmentation within a unified deep neural network:

$$(\hat{\mathbf{X}}, \hat{\mathbf{M}}) = f_{\theta}(\mathbf{X})$$

where:
* $\mathbf{X} \in \mathbb{R}^{B \times 1 \times H \times W}$ is the input noisy single-channel $\text{XCO}_2$ observation.
* $\hat{\mathbf{X}} \in \mathbb{R}^{B \times 1 \times H \times W}$ is the restored continuous $\text{XCO}_2$ concentration field (linear regression, no activation).
* $\hat{\mathbf{M}} \in [0, 1]^{B \times 1 \times H \times W}$ is the pixel-wise plume presence probability map ($\text{Sigmoid}$ activation).

```
                      +---------------------------------------+
                      |  Noisy Satellite Observation X        |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Shared Hierarchical ConvTransformer  |
                      |   Encoder (Stem, Down 1, Down 2)      |
                      +---------------------------------------+
                             |             |             |
                       F1 (32 ch)     F2 (64 ch)    F3 (128 ch)
                             |             |             |
            +----------------+             |             +-----------------+
            |                              |                               |
            v                              v                               v
+-----------------------+      +-----------------------+      +-----------------------+
|  Restoration Decoder  |      |  Feature Pyramid Net  |      |   Bottleneck (6x)     |
|  (Residual Upsampling)|      |  (Multi-Scale Fusion) |      | (Shifted Window Attn) |
+-----------------------+      +-----------------------+      +-----------------------+
            |                              |                               |
            v                              v                               |
+-----------------------+      +-----------------------+                   |
| Denoising Head (32->1)|      |  Recalibration Gating |                   |
| Continuous XCO2 Field |      |  Channel Sigmoid Attn |                   |
+-----------------------+      +-----------------------+                   |
                                           |                               |
                                           v                               |
                               +-----------------------+                   |
                               | Segmentation Head     |<------------------+
                               | Plume Probability Map |
                               +-----------------------+
```

---

## 2. Core Modules

### 2.1. ConvTransformer Block with Local Window Attention
Each ConvTransformer block consists of two sub-stages with residual skip connections:
1. **Window-based Multi-Head Self-Attention (W-MSA):** Local window partitioning reduces attention complexity from quadratic $\mathcal{O}((HW)^2)$ to linear $\mathcal{O}(M^2 HW)$ for window size $M \times M$.
2. **Relative Position Bias:**
   $$\mathbf{A} = \operatorname{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{B}_{\text{rel}}\right)\mathbf{V}$$
   where $\mathbf{B}_{\text{rel}} \in \mathbb{R}^{M^2 \times M^2}$ is drawn from a learnable parameter table $\hat{\mathbf{B}} \in \mathbb{R}^{(2M-1)^2 \times N_{\text{heads}}}$.
3. **Pointwise MLP:** Two $1 \times 1$ convolutions with $4\times$ channel expansion and GELU non-linearity.

### 2.2. Multi-Scale Feature Pyramid & Recalibration
The segmentation pathway extracts encoder features at three spatial scales:
* $\mathbf{F}_1 \in \mathbb{R}^{32 \times H \times W}$
* $\mathbf{F}_2 \in \mathbb{R}^{64 \times H/2 \times W/2}$
* $\mathbf{F}_3 \in \mathbb{R}^{128 \times H/4 \times W/4}$

1. **Lateral Projections:** $1 \times 1$ convolutions project each level to $256$ channels ($\mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3$).
2. **Top-Down Bilinear Fusion:**
   $$\tilde{\mathbf{P}}_2 = \mathbf{P}_2 + \uparrow_2 \mathbf{P}_3, \qquad \tilde{\mathbf{P}}_1 = \mathbf{P}_1 + \uparrow_2 \tilde{\mathbf{P}}_2$$
3. **Channel-wise Recalibration Gating:**
   $$\mathbf{g}_i = \operatorname{Sigmoid}(\operatorname{Conv}_{1\times1}(\operatorname{AdaptiveAvgPool}(\tilde{\mathbf{P}}_i)))$$
4. **Spatial Concatenation:** Each gated level is transformed through $3 \times 3$ convolutions to $128$ channels, spatially upsampled to match $\mathbf{F}_1$, and concatenated:
   $$\mathbf{F}_{\text{seg}} = \operatorname{Concat}[\mathbf{G}_1, \uparrow_2 \mathbf{G}_2, \uparrow_4 \mathbf{G}_3] \in \mathbb{R}^{384 \times H \times W}$$

---

## 3. Restricted Backbone Adaptation Protocol

During segmentation training, model parameters are partitioned into three distinct sets:
$$\theta = \{\theta_{\text{den}}^{\text{frozen}}, \theta_{\text{den}}^{\text{adapt}}, \theta_{\text{seg}}\}$$

* $\theta_{\text{den}}^{\text{frozen}}$: Pretrained denoiser core convolutional kernels and self-attention projections. During training, their gradients are zeroed:
  $$\nabla_{\theta_{\text{den}}^{\text{frozen}}} \mathcal{L}_{\text{total}} = 0$$
* $\theta_{\text{den}}^{\text{adapt}}$: All Group Normalization affine parameters and additive bias vectors. Kept trainable to calibrate internal activation distributions to the target region.
* $\theta_{\text{seg}}$: All Feature Pyramid Network and segmentation head parameters, trained end-to-end.

---

## 4. Multi-Task Joint Loss Formulation

The complete training criterion balances continuous concentration restoration and discrete plume segmentation:
$$\mathcal{L}_{\text{total}} = 0.3\,\mathcal{L}_{\text{den}} + 0.7\,\mathcal{L}_{\text{seg}}$$

### 4.1. Denoising Loss ($\mathcal{L}_{\text{den}}$)
$$\mathcal{L}_{\text{den}} = 0.8\,\mathcal{L}_{\text{MSE}} + 0.2\,(1 - \text{SSIM}) + 0.05\,\mathcal{L}_{\text{TV}} + 0.1\,\mathcal{L}_{\text{mass}}$$

* **Mean Squared Error:** Pixel-wise Euclidean distance between reconstructed field $\hat{\mathbf{X}}$ and clean target $X^{\text{clean}}$.
* **SSIM Loss:** Enforces structural similarity with a concentration dynamic range parameter $D_{\text{range}} = 6.0\,\text{ppm}$.
* **Total Variation Regularization:** Penalizes high-frequency noise oscillations:
  $$\mathcal{L}_{\text{TV}} = \frac{1}{B}\sum_{b=1}^B \left(\frac{1}{H(W-1)}\sum_{i,j}|\hat{X}_{i,j+1} - \hat{X}_{i,j}| + \frac{1}{(H-1)W}\sum_{i,j}|\hat{X}_{i+1,j} - \hat{X}_{i,j}|\right)$$
* **Plume Mass Conservation:**
  $$\mathcal{L}_{\text{mass}} = \frac{1}{B}\sum_{b=1}^B \left| \sum_{i,j} M_{i,j}\hat{X}_{i,j} - \sum_{i,j} M_{i,j}X^{\text{clean}}_{i,j} \right|$$

### 4.2. Boundary-Aware Physics Segmentation Loss ($\mathcal{L}_{\text{seg}}$)
$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{FT}} + 0.5\,\mathcal{L}_{\text{phys}} + 0.1\,\mathcal{L}_{\text{TV-mask}}$$

1. **Boundary-Weighted Focal Tversky Loss ($\mathcal{L}_{\text{FT}}$):**
   * Morphological dilation-erosion band extracts ambiguous boundary pixels:
     $$B_{\text{boundary}} = \operatorname{clamp}\left(\operatorname{MaxPool}_{5\times5}(\mathbf{Y}) - (-\operatorname{MaxPool}_{5\times5}(-\mathbf{Y})), 0, 1\right)$$
   * Spatial weight map: $w_{i,j} = 1.0 + 3.0 \cdot B_{\text{boundary}, i, j}$.
   * Tversky Index with $\alpha = 0.8$ (heavily penalizes false-alarm background activations):
     $$T = \frac{\sum p_{i,j} Y_{i,j} w_{i,j}}{\sum p_{i,j} Y_{i,j} w_{i,j} + \alpha \sum p_{i,j}(1-Y_{i,j})w_{i,j} + (1-\alpha)\sum (1-p_{i,j})Y_{i,j}w_{i,j} + 10^{-8}}$$
   * Focal scaling with $\gamma = 2.0$:
     $$\mathcal{L}_{\text{FT}} = (1 - T)^\gamma$$

2. **Atmospheric Physics Violation Constraint ($\mathcal{L}_{\text{phys}}$):**
   Anthropogenic plumes represent localized positive concentration anomalies over ambient baseline. Reconstructed pixels in predicted plume footprints ($p_{i,j} > 0.5$) falling below regional background ($c_{\text{thresh}} = 402.0\,\text{ppm}$) constitute physical violations:
   $$v_{i,j} = \min\left(15.0, \max\left(0, c_{\text{thresh}} - X_{\text{phys}, i, j}\right) \cdot \mathbb{I}(p_{i,j} > 0.5)\right)$$
   Stabilized exponential-logarithmic penalty with temperature $T_{\text{temp}} = 8.0$:
   $$\mathcal{L}_{\text{phys}} = \frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W \log(1 + v_{i,j}) \cdot \exp\left(\frac{v_{i,j}}{T_{\text{temp}}}\right)$$

3. **Mask Total Variation Regularization ($\mathcal{L}_{\text{TV-mask}}$):**
   Suppresses isolated pixel noise and enforces continuous, smooth plume boundary contours.

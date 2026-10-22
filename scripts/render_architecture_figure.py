import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import os

def render_figure():
    fig = plt.figure(figsize=(26, 13.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 52)
    ax.axis('off')

    # Professional Theme Colors
    C_BG_ENC = '#F4F7FB'
    C_BORDER_ENC = '#2B6CB0'
    C_BG_DEC = '#FDF8F3'
    C_BORDER_DEC = '#C05621'
    C_BG_SEG = '#F2F9F5'
    C_BORDER_SEG = '#276749'
    C_BG_LOSS = '#FCF4F6'
    C_BORDER_LOSS = '#97266D'
    
    C_STEM = '#FEEBC8'
    C_STAGE = '#EBF8FF'
    C_BOTTLENECK = '#BEE3F8'
    C_DEC_STAGE = '#C6F6D5'
    C_HEAD = '#EDF2F7'
    C_FPN = '#E9D8FD'
    C_SEG_HEAD = '#FEEBC8'
    C_PHYS = '#FEFCBF'
    C_JOINT = '#1A202C'

    def draw_box(x, y, w, h, bg, border, lw=1.5, ls='-', radius=0.6, zorder=2):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.15,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#2D3748', lw=2.0, ls='-', zorder=5):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, ls=ls,
                mutation_scale=13, shrinkA=2, shrinkB=2
            ),
            zorder=zorder
        )

    def draw_path(pts, color='#2D3748', lw=2.0, ls='-', zorder=5):
        for i in range(len(pts) - 1):
            if i == len(pts) - 2:
                draw_arrow(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], color=color, lw=lw, ls=ls, zorder=zorder)
            else:
                ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]], color=color, lw=lw, ls=ls, zorder=zorder)

    # Title Banner
    ax.text(52, 50.4, "Proposed Joint Denoising–Segmentation Framework Architecture", 
            ha='center', va='center', fontsize=21, fontweight='bold', color='#1A202C')

    # ==================== 1. SHARED HIERARCHICAL ENCODER CONTAINER ====================
    # Increased height to 21.2 to give ample space for skip connections
    draw_box(13.5, 27.2, 43.5, 21.6, C_BG_ENC, C_BORDER_ENC, lw=2.0, ls='--', radius=1.0, zorder=1)
    ax.text(35.25, 47.6, "Shared Hierarchical Encoder (EfficientDenoiser Backbone)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#2B6CB0', zorder=3)
    ax.text(35.25, 46.2, "Pretrained Backbone: Frozen Weights (θ_den^(frozen), ∇=0)  |  Trainable Norm & Bias (θ_den^(adapt))", 
            ha='center', va='center', fontsize=9.2, fontweight='semibold', color='#4A5568', zorder=3)

    # ==================== 2. DENOISING DECODER CONTAINER ====================
    draw_box(58.5, 27.2, 30.5, 21.6, C_BG_DEC, C_BORDER_DEC, lw=2.0, ls='--', radius=1.0, zorder=1)
    ax.text(73.75, 47.6, "Denoising Decoder (Auxiliary Reconstruction Pathway)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#C05621', zorder=3)
    ax.text(73.75, 46.2, "Continuous Concentration Field Regression  |  Linear Output (No Activation)", 
            ha='center', va='center', fontsize=9.2, fontweight='semibold', color='#7B341E', zorder=3)

    # ==================== 3. MULTI-SCALE SEGMENTATION CONTAINER ====================
    draw_box(13.5, 5.0, 58.5, 20.8, C_BG_SEG, C_BORDER_SEG, lw=2.0, ls='--', radius=1.0, zorder=1)
    ax.text(42.75, 24.5, "Multi-Scale Segmentation Pathway (FPN + Recalibration + Segmentation Head)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#276749', zorder=3)
    ax.text(42.75, 23.2, "Fully Trainable Modules (θ_seg)  |  Reuses Multi-Scale Encoder Representations (F₁, F₂, F₃)", 
            ha='center', va='center', fontsize=9.2, fontweight='semibold', color='#22543D', zorder=3)

    # ==================== 4. PHYSICS & JOINT LOSS CONTAINER ====================
    draw_box(73.5, 5.0, 29.0, 20.8, C_BG_LOSS, C_BORDER_LOSS, lw=2.0, ls='--', radius=1.0, zorder=1)
    ax.text(88.0, 24.5, "Physics-Constrained Joint Loss & Optimization", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#97266D', zorder=3)
    ax.text(88.0, 23.2, "Joint Multi-Task Criterion with Physics-Informed Concentration Coupling", 
            ha='center', va='center', fontsize=9.2, fontweight='semibold', color='#702459', zorder=3)

    # ==================== INPUT IMAGE BOX ====================
    draw_box(1.2, 30.5, 10.5, 13.0, '#FFFFFF', '#A0AEC0', lw=1.5, radius=0.6, zorder=3)
    ax.text(6.45, 42.1, "Noisy XCO₂ Input X", ha='center', va='center', fontsize=11, fontweight='bold', color='#1A202C', zorder=4)
    ax.text(6.45, 40.8, "(B × 1 × H × W)", ha='center', va='center', fontsize=8.8, fontweight='semibold', color='#718096', zorder=4)
    
    noisy_thumb = 'Paper/final_thumb_noisy.png'
    if os.path.exists(noisy_thumb):
        im_in = Image.open(noisy_thumb)
        box_in = OffsetImage(im_in, zoom=0.38)
        ab_in = AnnotationBbox(box_in, (6.45, 35.7), frameon=True, pad=0.1, 
                               bboxprops=dict(edgecolor='#CBD5E0', lw=1.2), zorder=4)
        ax.add_artist(ab_in)
    ax.text(6.45, 31.6, "Satellite Observation", ha='center', va='center', fontsize=8.5, color='#4A5568', zorder=4)

    draw_arrow(11.7, 36.0, 15.0, 36.0, color='#2D3748', lw=2.2)

    # ==================== SHARED ENCODER STAGES ====================
    # Lowered slightly to y=30.0 so top space is clear
    draw_box(15.0, 30.0, 8.2, 9.8, C_STEM, '#DD6B20', lw=1.8, radius=0.6, zorder=3)
    ax.text(19.1, 38.3, "Stem", ha='center', va='center', fontsize=11, fontweight='bold', color='#7B341E', zorder=4)
    ax.text(19.1, 36.0, "Conv 7×7 (s=1, p=3)\nGroupNorm (8, 32)\nGELU Activation", 
            ha='center', va='center', fontsize=8.2, color='#2D3748', zorder=4)
    ax.text(19.1, 31.2, "Output F₁ (x₀)\n32 × H × W", ha='center', va='center', 
            fontsize=8.8, fontweight='bold', color='#2B6CB0', zorder=4)

    draw_box(25.0, 30.0, 9.2, 9.8, C_STAGE, '#3182CE', lw=1.8, radius=0.6, zorder=3)
    ax.text(29.6, 38.3, "Downsample Stage 1", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2B6CB0', zorder=4)
    ax.text(29.6, 35.8, "Strided Conv 3×3 (s=2)\n2× ConvTransformer\n(dim=64, heads=4, ws=8)", 
            ha='center', va='center', fontsize=8.0, color='#2D3748', zorder=4)
    ax.text(29.6, 31.2, "Output F₂ (x₁)\n64 × H/2 × W/2", ha='center', va='center', 
            fontsize=8.8, fontweight='bold', color='#2B6CB0', zorder=4)

    draw_box(36.0, 30.0, 9.2, 9.8, C_STAGE, '#3182CE', lw=1.8, radius=0.6, zorder=3)
    ax.text(40.6, 38.3, "Downsample Stage 2", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2B6CB0', zorder=4)
    ax.text(40.6, 35.8, "Strided Conv 3×3 (s=2)\n2× ConvTransformer\n(dim=128, heads=8, ws=4)", 
            ha='center', va='center', fontsize=8.0, color='#2D3748', zorder=4)
    ax.text(40.6, 31.2, "Output F₃ (x₂)\n128 × H/4 × W/4", ha='center', va='center', 
            fontsize=8.8, fontweight='bold', color='#2B6CB0', zorder=4)

    draw_box(47.0, 30.0, 8.8, 9.8, C_BOTTLENECK, '#2B6CB0', lw=2.0, radius=0.6, zorder=3)
    ax.text(51.4, 38.3, "Bottleneck", ha='center', va='center', fontsize=11, fontweight='bold', color='#1A365D', zorder=4)
    ax.text(51.4, 35.5, "6× ConvTransformer\n(dim=128, heads=8, ws=4)\nShifted-Window Attn\n(shift alternating 2, 0)", 
            ha='center', va='center', fontsize=7.8, color='#1A202C', zorder=4)
    ax.text(51.4, 31.2, "Output x_b\n128 × H/4 × W/4", ha='center', va='center', 
            fontsize=8.8, fontweight='bold', color='#1A365D', zorder=4)

    draw_arrow(23.2, 35.0, 25.0, 35.0, color='#2D3748', lw=2.0)
    draw_arrow(34.2, 35.0, 36.0, 35.0, color='#2D3748', lw=2.0)
    draw_arrow(45.2, 35.0, 47.0, 35.0, color='#2D3748', lw=2.0)

    # ==================== DENOISING DECODER ====================
    draw_arrow(55.8, 35.0, 59.5, 35.0, color='#2D3748', lw=2.0)

    draw_box(59.5, 30.0, 8.8, 9.8, C_DEC_STAGE, '#2F855A', lw=1.8, radius=0.6, zorder=3)
    ax.text(63.9, 38.3, "Decoder Stage 1", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#22543D', zorder=4)
    ax.text(63.9, 35.8, "Nearest Upsample (2×)\nConv 3×3 (128→64)\n2× ConvTransformer\n(dim=64, heads=4, ws=8)", 
            ha='center', va='center', fontsize=7.8, color='#2D3748', zorder=4)
    ax.text(63.9, 31.2, "⊕ Add Skip from F₂\n64 × H/2 × W/2", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color='#2B6CB0', zorder=4)

    draw_box(70.0, 30.0, 8.8, 9.8, C_DEC_STAGE, '#2F855A', lw=1.8, radius=0.6, zorder=3)
    ax.text(74.4, 38.3, "Decoder Stage 2", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#22543D', zorder=4)
    ax.text(74.4, 35.8, "Nearest Upsample (2×)\nConv 3×3 (64→32)\n2× ConvTransformer\n(dim=32, heads=2, ws=16)", 
            ha='center', va='center', fontsize=7.8, color='#2D3748', zorder=4)
    ax.text(74.4, 31.2, "⊕ Add Skip from F₁\n32 × H × W", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color='#2B6CB0', zorder=4)

    draw_box(80.5, 30.5, 7.2, 8.8, C_HEAD, '#718096', lw=1.8, radius=0.6, zorder=3)
    ax.text(84.1, 37.1, "Denoising Head", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2D3748', zorder=4)
    ax.text(84.1, 34.0, "Conv 3×3 (32→1, p=1)\nLinear Regression\n(No Activation)\nCUDA FP16 Autocast", 
            ha='center', va='center', fontsize=8.0, color='#4A5568', zorder=4)

    draw_arrow(68.3, 35.0, 70.0, 35.0, color='#2D3748', lw=2.0)
    draw_arrow(78.8, 35.0, 80.5, 35.0, color='#2D3748', lw=2.0)

    # Denoised Output Box
    draw_box(89.8, 29.5, 10.5, 13.0, '#FFFFFF', '#A0AEC0', lw=1.5, radius=0.6, zorder=3)
    ax.text(95.05, 41.1, "Denoised Field X̂", ha='center', va='center', fontsize=11, fontweight='bold', color='#1A202C', zorder=4)
    ax.text(95.05, 39.8, "(B × 1 × H × W)", ha='center', va='center', fontsize=8.8, fontweight='semibold', color='#718096', zorder=4)
    
    den_thumb = 'Paper/final_thumb_denoised.png'
    if os.path.exists(den_thumb):
        im_den = Image.open(den_thumb)
        box_den = OffsetImage(im_den, zoom=0.38)
        ab_den = AnnotationBbox(box_den, (95.05, 34.7), frameon=True, pad=0.1, 
                                bboxprops=dict(edgecolor='#CBD5E0', lw=1.2), zorder=4)
        ax.add_artist(ab_den)
    ax.text(95.05, 30.6, "Continuous XCO₂", ha='center', va='center', fontsize=8.5, color='#4A5568', zorder=4)

    draw_arrow(87.7, 35.0, 89.8, 35.0, color='#2D3748', lw=2.2)

    # ==================== SKIP CONNECTIONS (Elevated with ample clearance) ====================
    # Skip F2 -> Decoder Stage 1 at y = 42.0
    draw_path([(29.6, 39.8), (29.6, 42.0), (63.9, 42.0), (63.9, 39.8)], color='#2B6CB0', lw=2.0, ls='-', zorder=5)
    draw_box(39.0, 41.2, 16.5, 1.6, '#FFFFFF', '#2B6CB0', lw=1.0, radius=0.3, zorder=6)
    ax.text(47.25, 42.0, "Skip: Add F₂ (64 × H/2 × W/2)", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color='#2B6CB0', zorder=7)

    # Skip F1 -> Decoder Stage 2 at y = 44.2
    draw_path([(19.1, 39.8), (19.1, 44.2), (74.4, 44.2), (74.4, 39.8)], color='#2B6CB0', lw=2.0, ls='-', zorder=5)
    draw_box(39.0, 43.4, 16.5, 1.6, '#FFFFFF', '#2B6CB0', lw=1.0, radius=0.3, zorder=6)
    ax.text(47.25, 44.2, "Skip: Add F₁ (32 × H × W)", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color='#2B6CB0', zorder=7)

    # ==================== MULTI-SCALE SEGMENTATION PATHWAY ====================
    # FPN Sub-frame
    draw_box(14.5, 6.0, 36.5, 16.8, '#FFFFFF', '#805AD5', lw=1.2, ls=':', radius=0.6, zorder=2)
    ax.text(32.75, 22.0, "Feature Pyramid Network (FPN) & Channel Gating Recalibration", 
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#553C9A', zorder=3)

    # Laterals
    draw_box(15.5, 17.2, 8.8, 3.8, C_FPN, '#805AD5', lw=1.2, radius=0.4, zorder=3)
    ax.text(19.9, 19.1, "Lateral Conv 1×1 (32→256)\n→ P₁ (256 × H × W)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#44337A', zorder=4)

    draw_box(15.5, 12.3, 8.8, 3.8, C_FPN, '#805AD5', lw=1.2, radius=0.4, zorder=3)
    ax.text(19.9, 14.2, "Lateral Conv 1×1 (64→256)\n→ P₂ (256 × H/2 × W/2)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#44337A', zorder=4)

    draw_box(15.5, 7.4, 8.8, 3.8, C_FPN, '#805AD5', lw=1.2, radius=0.4, zorder=3)
    ax.text(19.9, 9.3, "Lateral Conv 1×1 (128→256)\n→ P₃ (256 × H/4 × W/4)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#44337A', zorder=4)

    # Clear, uncrossed feature routing from encoder to laterals
    # F1 -> P1: straight drop
    draw_arrow(19.1, 30.0, 19.1, 21.0, color='#805AD5', lw=1.8)
    # F2 -> P2: drop down and into P2
    draw_path([(29.6, 30.0), (29.6, 26.0), (14.0, 26.0), (14.0, 14.2), (15.5, 14.2)], color='#805AD5', lw=1.8, ls='-', zorder=5)
    # F3 -> P3: drop down and into P3
    draw_path([(40.6, 30.0), (40.6, 26.8), (13.6, 26.8), (13.6, 9.3), (15.5, 9.3)], color='#805AD5', lw=1.8, ls='-', zorder=5)

    # Top-Down Fusion & Channel Gating Box
    draw_box(26.5, 7.6, 11.5, 13.8, C_FPN, '#805AD5', lw=1.5, radius=0.5, zorder=3)
    ax.text(32.25, 20.4, "Top-Down Fusion & Gating", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#322659', zorder=4)
    ax.text(32.25, 17.5, "Top-Down Aggregation:\nP̃₂ = P₂ + ↑₂(P₃)\nP̃₁ = P₁ + ↑₂(P̃₂)", 
            ha='center', va='center', fontsize=8.2, fontweight='semibold', color='#2D3748', zorder=4)
    ax.text(32.25, 12.0, "Channel-wise Recalibration:\nAdaptiveAvgPool → Conv 1×1 → σ\n\nGated Feature:\nP̃_i ⊙ Gate(P̃_i)\n(Independent Per-Level Recalibration)", 
            ha='center', va='center', fontsize=7.8, color='#4A5568', zorder=4)

    draw_arrow(24.3, 19.1, 26.5, 17.5, color='#805AD5', lw=1.5)
    draw_arrow(24.3, 14.2, 26.5, 14.2, color='#805AD5', lw=1.5)
    draw_arrow(24.3, 9.3, 26.5, 11.0, color='#805AD5', lw=1.5)

    # Multi-Scale Fusion Convolutions & Alignment Box
    draw_box(40.2, 7.6, 10.0, 13.8, C_FPN, '#6B46C1', lw=1.8, radius=0.5, zorder=3)
    ax.text(45.2, 20.4, "Multi-Scale Fusion & Concat", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#322659', zorder=4)
    ax.text(45.2, 16.5, "Fusion Convolutions:\nConv 3×3 (256→128)\n+ GroupNorm(8) + GELU\nYields G₁, G₂, G₃ (128 ch)", 
            ha='center', va='center', fontsize=7.8, color='#2D3748', zorder=4)
    ax.text(45.2, 11.5, "Spatial Alignment:\nBilinear Upsample ↑₂G₂, ↑₄G₃\n\nChannel Concatenation:\nF_seg = [G₁, ↑₂G₂, ↑₄G₃]\n→ 384 × H × W", 
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#2D3748', zorder=4)

    draw_arrow(38.0, 14.5, 40.2, 14.5, color='#6B46C1', lw=2.0)

    # Segmentation Head Sub-frame
    draw_box(52.2, 6.0, 18.8, 16.8, '#FFFFFF', '#DD6B20', lw=1.2, ls=':', radius=0.6, zorder=2)
    ax.text(61.6, 22.0, "Segmentation Head Architecture", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#C05621', zorder=3)

    # Block 1: ConvTransformer
    draw_box(53.2, 17.0, 8.0, 4.0, C_SEG_HEAD, '#DD6B20', lw=1.5, radius=0.4, zorder=3)
    ax.text(57.2, 19.0, "ConvTransformer Block\n(384 ch, heads=12, ws=8)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#7B341E', zorder=4)

    # Block 2: Grouped Conv
    draw_box(53.2, 11.8, 8.0, 4.2, C_SEG_HEAD, '#DD6B20', lw=1.5, radius=0.4, zorder=3)
    ax.text(57.2, 13.9, "Grouped Conv 3×3\n(384 ch, groups=128)\n+ GELU Activation", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#7B341E', zorder=4)

    # Block 3: Progressive Channel Reduction
    draw_box(53.2, 6.6, 8.0, 4.2, C_SEG_HEAD, '#DD6B20', lw=1.5, radius=0.4, zorder=3)
    ax.text(57.2, 8.7, "Gradual Reduction:\nConv 1×1 (384→256)\nConv 3×3 (256→128, p=1)\nConv 1×1 (128→1) + Sigmoid", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#7B341E', zorder=4)

    # Stepped connector from FPN into Segmentation Head
    draw_path([(50.2, 14.5), (51.7, 14.5), (51.7, 19.0), (53.2, 19.0)], color='#DD6B20', lw=2.0, ls='-', zorder=5)
    draw_arrow(57.2, 17.0, 57.2, 16.0, color='#DD6B20', lw=2.0)
    draw_arrow(57.2, 11.8, 57.2, 10.8, color='#DD6B20', lw=2.0)

    # Plume Probability Map Output Box
    draw_box(62.4, 7.8, 8.0, 11.5, '#FFFFFF', '#A0AEC0', lw=1.5, radius=0.6, zorder=3)
    ax.text(66.4, 18.0, "Plume Map M̂", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1A202C', zorder=4)
    ax.text(66.4, 16.9, "Probability Map ∈ [0, 1]", ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#718096', zorder=4)
    
    mask_thumb = 'Paper/final_thumb_mask.png'
    if os.path.exists(mask_thumb):
        im_mask = Image.open(mask_thumb)
        box_mask = OffsetImage(im_mask, zoom=0.36)
        ab_mask = AnnotationBbox(box_mask, (66.4, 13.0), frameon=True, pad=0.1, 
                                 bboxprops=dict(edgecolor='#CBD5E0', lw=1.2), zorder=4)
        ax.add_artist(ab_mask)
    ax.text(66.4, 9.0, "Binary Plume Region", ha='center', va='center', fontsize=8.2, color='#4A5568', zorder=4)

    draw_arrow(61.2, 8.7, 62.4, 13.0, color='#DD6B20', lw=2.0)

    # ==================== PHYSICS & JOINT LOSS ====================
    # Denoising Loss Box
    draw_box(74.5, 17.5, 27.0, 4.4, '#FED7D7', '#E53E3E', lw=1.6, radius=0.4, zorder=3)
    ax.text(88.0, 20.5, "Denoising Objective L_den (PhysicsSSIMLoss)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#9B2C2C', zorder=4)
    ax.text(88.0, 18.7, "L_den = 0.8 MSE + 0.2 (1-SSIM) + 0.05 TV + 0.1 L_mass\n• Evaluated vs clean target X^clean  • L_mass enforces mass conservation on M", 
            ha='center', va='center', fontsize=7.8, color='#1A202C', zorder=4)

    # Physics Coupling Module Box (Central Highlight!)
    draw_box(74.5, 11.6, 27.0, 5.0, C_PHYS, '#D69E2E', lw=2.2, radius=0.4, zorder=3)
    ax.text(88.0, 15.2, "⚡ Physics Coupling & Consistency Constraint (L_phys)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#744210', zorder=4)
    ax.text(88.0, 13.2, "Couples Denoised Field X̂ with Predicted Plume Mask M̂:\n• Physical Conversion: X_phys = 3.42 X̂ + 402.0 ppm\n• Constraint: Predicted Plume (M̂ > 0.5) must satisfy X_phys ≥ 402.0 ppm\n• Violation = ReLU(402.0 - X_phys ⊙ M̂) clamped & penalized via exp-log scaling", 
            ha='center', va='center', fontsize=7.6, color='#2D3748', zorder=4)

    # Segmentation Loss Box
    draw_box(74.5, 6.4, 27.0, 4.4, '#FED7D7', '#E53E3E', lw=1.6, radius=0.4, zorder=3)
    ax.text(88.0, 9.4, "Segmentation Objective L_seg (CO2PlumeSegmentationLoss)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#9B2C2C', zorder=4)
    ax.text(88.0, 7.6, "L_seg = L_FT + 0.5 L_phys + 0.1 L_TV-mask\n• Boundary-weighted Focal Tversky (α=0.8, γ=2, band w=3.0)  • L_phys coupling", 
            ha='center', va='center', fontsize=7.8, color='#1A202C', zorder=4)

    # Joint Objective Box (Slate Dark Banner)
    draw_box(74.5, 1.2, 27.0, 4.4, C_JOINT, '#1A202C', lw=2.0, radius=0.4, zorder=3)
    ax.text(88.0, 4.0, "Joint Optimization Objective: L_total = 0.3 L_den + 0.7 L_seg", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF', zorder=4)
    ax.text(88.0, 2.3, "Gradient Updates: Trainable θ_seg & Adapted θ_den^(adapt) (Norm/Bias) | Frozen: θ_den^(frozen) (∇=0)", 
            ha='center', va='center', fontsize=8.0, fontweight='semibold', color='#CBD5E0', zorder=4)

    # Arrows to Loss (Clean exterior routes)
    # Denoised Output -> L_den
    draw_path([(95.05, 29.5), (95.05, 21.9)], color='#E53E3E', lw=2.0, ls='--', zorder=5)
    # Denoised Output -> Physics Coupling (Routes along right border)
    draw_path([(98.0, 29.5), (98.0, 16.6), (97.0, 16.6)], color='#D69E2E', lw=2.2, ls='--', zorder=5)
    
    # Plume Output -> Physics Coupling
    draw_path([(70.4, 14.1), (74.5, 14.1)], color='#D69E2E', lw=2.2, ls='--', zorder=5)
    # Plume Output -> L_seg
    draw_path([(70.4, 8.6), (74.5, 8.6)], color='#E53E3E', lw=2.0, ls='--', zorder=5)
    
    # Physics -> L_seg
    draw_arrow(88.0, 11.6, 88.0, 10.8, color='#D69E2E', lw=2.0, ls='-')
    # L_den -> Joint
    draw_path([(75.5, 17.5), (75.5, 5.6)], color='#E53E3E', lw=2.0, ls='--', zorder=5)
    # L_seg -> Joint
    draw_arrow(88.0, 6.4, 88.0, 5.6, color='#E53E3E', lw=2.0, ls='--')

    # ==================== LEGEND ====================
    draw_box(13.5, 0.8, 58.5, 3.6, '#FFFFFF', '#CBD5E0', lw=1.2, radius=0.4, zorder=3)
    ax.text(14.8, 2.6, "LEGEND:", fontsize=9.5, fontweight='bold', color='#1A202C', va='center', zorder=4)
    
    # Feature Flow
    draw_arrow(19.8, 2.6, 23.3, 2.6, color='#2D3748', lw=2.0)
    ax.text(23.8, 2.6, "Feature Flow", fontsize=8.5, color='#2D3748', va='center', zorder=4)

    # Skip Connection
    draw_arrow(31.2, 2.6, 34.7, 2.6, color='#2B6CB0', lw=2.0)
    ax.text(35.2, 2.6, "Additive Skip", fontsize=8.5, color='#2B6CB0', va='center', zorder=4)

    # Loss Flow
    draw_arrow(43.2, 2.6, 46.7, 2.6, color='#E53E3E', lw=2.0, ls='--')
    ax.text(47.2, 2.6, "Loss / Supervision", fontsize=8.5, color='#C53030', va='center', zorder=4)

    # Physics Coupling
    draw_arrow(57.2, 2.6, 60.7, 2.6, color='#D69E2E', lw=2.2, ls='--')
    ax.text(61.2, 2.6, "Physics Coupling", fontsize=8.5, color='#B7791F', va='center', zorder=4)

    # Parameter Status Badges in Legend
    draw_box(14.8, 1.1, 16.5, 1.2, '#EBF8FF', '#3182CE', lw=1.0, radius=0.2, zorder=4)
    ax.text(23.05, 1.7, "Pretrained Frozen Backbone (∇=0)", fontsize=7.6, fontweight='bold', color='#2B6CB0', ha='center', va='center', zorder=5)

    draw_box(32.8, 1.1, 17.5, 1.2, '#FEFCBF', '#D69E2E', lw=1.0, radius=0.2, zorder=4)
    ax.text(41.55, 1.7, "Adapted Parameters (Norm/Bias)", fontsize=7.6, fontweight='bold', color='#744210', ha='center', va='center', zorder=5)

    draw_box(51.8, 1.1, 19.0, 1.2, '#F0FFF4', '#38A169', lw=1.0, radius=0.2, zorder=4)
    ax.text(61.3, 1.7, "Fully Trainable Modules (FPN & Seg Head)", fontsize=7.6, fontweight='bold', color='#22543D', ha='center', va='center', zorder=5)

    plt.tight_layout()
    output_png = 'Paper/1.png'
    output_png2 = 'Paper/architecture.png'
    output_pdf = 'Paper/architecture.pdf'
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_png2, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    print("Clean perfected figure rendered successfully!")

if __name__ == '__main__':
    render_figure()

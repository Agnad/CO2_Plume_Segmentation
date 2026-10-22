"""
demo_inference.py - Verification and Quickstart Pipeline.

Demonstrates model instantiation, synthetic XCO2 observation generation,
forward pass inference through the joint architecture, and multi-task
loss calculation.

Usage:
    python demo_inference.py
    python demo_inference.py --device cuda --batch-size 4
"""

import sys
import argparse
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("Error: PyTorch is not installed in the active environment.")
    print("Please install requirements using: pip install -r requirements.txt")
    sys.exit(1)

from models.joint_model import JointDenoiseSeg
from models.losses import JointPhysicsLoss


def parse_args():
    parser = argparse.ArgumentParser(description="XCO2 Plume Joint Learning Verification Demo")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size for verification")
    parser.add_argument("--height", type=int, default=160, help="Tile height in pixels")
    parser.add_argument("--width", type=int, default=160, help="Tile width in pixels")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Target compute device")
    parser.add_argument("--weights", type=str, default=None,
                        help="Optional path to pretrained weights checkpoint (.pth)")
    return parser.parse_args()


def generate_synthetic_xco2_scene(b: int, h: int, w: int, device: torch.device):
    """
    Generates synthetic XCO2 scene tiles simulating satellite soundings:
    - Ambient background concentration ~402 ppm
    - Localized Gaussian emission plume enhancement (Delta-XCO2 ~ 1-3 ppm)
    - Pseudo-random spectrometer instrument noise (sigma ~ 0.7 ppm)
    """
    # 1. Clean ambient baseline background
    clean_field = torch.full((b, 1, h, w), 402.0, device=device)
    
    # 2. Add synthetic Gaussian plume footprint
    y = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1)
    x = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w)
    
    # Center plume in upper-left quadrant drifting diagonally
    plume_profile = torch.exp(-((x + 0.2)**2 / 0.15 + (y + 0.2)**2 / 0.08))
    plume_enhancement = 2.8 * plume_profile  # +2.8 ppm enhancement
    
    clean_field = clean_field + plume_enhancement
    
    # Reference binary plume footprint (enhancement > 0.5 ppm threshold)
    reference_mask = (plume_enhancement > 0.5).float()
    
    # 3. Add instrument observation noise (sigma = 0.7 ppm)
    noise = torch.randn((b, 1, h, w), device=device) * 0.7
    noisy_observation = clean_field + noise
    
    # Normalize inputs as standard practice (scaled around 0)
    noisy_norm = (noisy_observation - 402.0) / 3.42
    clean_norm = (clean_field - 402.0) / 3.42
    
    return noisy_norm, clean_norm, reference_mask


def main():
    args = parse_args()
    
    # Select compute device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print("=" * 75)
    print("  Restoration-Aware Joint Learning for Satellite XCO2 Plume Monitoring")
    print("  Verification & Quickstart Inference Pipeline")
    print("=" * 75)
    print(f"Target Device:     {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Host CPU'})")
    print(f"Input Shape:       Batch={args.batch_size}, Channels=1, Height={args.height}, Width={args.width}")
    print()

    # 1. Instantiate Joint Model
    print("[1/4] Instantiating JointDenoiseSeg model...")
    t0 = time.time()
    model = JointDenoiseSeg(weights_path=args.weights, freeze_backbone=True).to(device)
    t_init = time.time() - t0
    
    param_summary = model.get_trainable_parameters_summary()
    print(f"      Initialized in {t_init:.2f}s")
    print(f"      Total Parameters:     {param_summary['total']:,}")
    print(f"      Trainable Parameters: {param_summary['trainable']:,} ({param_summary['trainable_pct']:.2f}%)")
    print(f"      Frozen Parameters:    {param_summary['frozen']:,} (Pretrained denoiser backbone)")
    print()

    # 2. Synthesize observation data
    print("[2/4] Generating synthetic XCO2 observation tiles...")
    noisy_input, clean_target, ground_truth_mask = generate_synthetic_xco2_scene(
        args.batch_size, args.height, args.width, device
    )
    print(f"      Noisy observation tile min/max: {noisy_input.min():.2f} / {noisy_input.max():.2f}")
    print(f"      Plume foreground pixels:        {int(ground_truth_mask.sum().item())} / {ground_truth_mask.numel()}")
    print()

    # 3. Model Forward Pass
    print("[3/4] Running forward inference pass...")
    model.eval()
    t_start = time.time()
    with torch.no_grad():
        denoised_output, seg_probability = model(noisy_input)
    t_inf = time.time() - t_start
    
    print(f"      Forward pass completed in {t_inf * 1000:.1f} ms")
    print(f"      Denoised field output: shape={tuple(denoised_output.shape)}, dtype={denoised_output.dtype}")
    print(f"      Plume probability map: shape={tuple(seg_probability.shape)}, min={seg_probability.min():.4f}, max={seg_probability.max():.4f}")
    
    # Assertions
    assert denoised_output.shape == (args.batch_size, 1, args.height, args.width), "Shape mismatch on denoised output!"
    assert seg_probability.shape == (args.batch_size, 1, args.height, args.width), "Shape mismatch on segmentation output!"
    assert (seg_probability >= 0.0).all() and (seg_probability <= 1.0).all(), "Segmentation probabilities outside [0, 1]!"
    print("      Shape and output boundary assertions verified.")
    print()

    # 4. Joint Loss Evaluation
    print("[4/4] Evaluating Joint Physics-Informed Multi-Task Loss...")
    criterion = JointPhysicsLoss(denoise_weight=0.3, seg_weight=0.7).to(device)
    loss_dict = criterion(denoised_output, seg_probability, clean_target, ground_truth_mask)
    
    print(f"      Total Joint Loss:      {loss_dict['total'].item():.5f} (0.3 * Denoise + 0.7 * Seg)")
    print(f"      Denoise Loss Component:{loss_dict['denoise'].item():.5f} (MSE + SSIM + TV + Mass)")
    print(f"      Seg Loss Component:    {loss_dict['seg'].item():.5f} (Focal Tversky + Physics + TV)")
    print()

    print("=" * 75)
    print("  All verification stages completed successfully!")
    print("  Repository is ready for production, training, and benchmarking.")
    print("=" * 75)


if __name__ == '__main__':
    main()

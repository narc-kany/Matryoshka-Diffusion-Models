"""
Progressive multi-scale inpainting demo (concept + runnable skeleton).
- Provide your MDM model wrapper (see notes below)
- This script runs coarse->fine inpainting by resizing the image+mask,
  performing inpainting at each scale and upsampling to the next finer scale.
"""

import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils import load_image_as_tensor, save_tensor_as_image, load_mask_as_tensor

# ---- Replace / implement these to call the real MDM sampling API ----
# The functions are placeholders so the script is runnable structure-wise.
# If you use Apple ml-mdm, implement these to call their sample/inpaint methods.
def load_mdm_model(ckpt_path, device):
    """
    Load a Matryoshka Diffusion Model (or compatible diffusion model).
    Return an object with a method `inpaint(image, mask, steps, guidance)` or
    similar. The exact API depends on which implementation you use.
    """
    # Example placeholder: return a dummy object (NOT real inference)
    class DummyModel:
        def __init__(self, device):
            self.device = device
        @torch.no_grad()
        def inpaint(self, image, mask, steps=50):
            # identity: returns input (replace with real sampling)
            return image
    return DummyModel(device)

def inpaint_one_scale(model, image_tensor, mask_tensor, steps=50, device='cuda'):
    """
    Run inpainting/sampling at the given scale.
    - image_tensor: torch tensor shape [C,H,W], values in [-1,1] or [0,1] depending on your model
    - mask_tensor: torch tensor shape [1,H,W], 1 = masked region to fill, 0 = keep
    Return: inpainted image tensor same shape as image_tensor
    """
    # Convert to batch dims
    x = image_tensor.unsqueeze(0).to(device)
    m = mask_tensor.unsqueeze(0).to(device)
    # Replace below with actual model inpaint call
    out = model.inpaint(x, m, steps=steps)
    # ensure shape and detach
    return out.squeeze(0).cpu()

# ----------------------------------------------------------------------

def progressive_inpaint(model, orig_img_path, orig_mask_path, out_dir,
                        scales=[128, 256, 512], steps_per_scale=50, device='cuda'):
    """
    Progressive inpainting:
      - For each scale (coarse->fine):
        1. Resize original image and mask to that scale
        2. Run model.inpaint at that scale
        3. Upsample the inpainted result to the next scale and use as init
    """
    os.makedirs(out_dir, exist_ok=True)
    # load original image & mask as tensors (C,H,W) in [0,1]
    img0 = load_image_as_tensor(orig_img_path)  # [C,H,W], float32 0..1
    mask0 = load_mask_as_tensor(orig_mask_path) # [1,H,W], binary 0/1

    prev_inpaint = None
    for i, s in enumerate(scales):
        print(f"\n--- Scale {s} ---")
        # resize image + mask to scale s (preserve aspect by center crop/rescale)
        img_s = TF.resize(img0, [s, s])
        mask_s = TF.resize(mask0, [s, s], interpolation=Image.NEAREST)

        # if we have a previous inpaint result, upsample to current scale and paste
        if prev_inpaint is not None:
            prev_up = TF.resize(prev_inpaint, [s, s])
            # combine: keep original unmasked pixels, replace masked with prev_up (warm start)
            img_s = img_s * (1 - mask_s) + prev_up * mask_s

        # convert to model input range if needed (example: [-1,1])
        img_input = img_s * 2.0 - 1.0  # convert 0..1 -> -1..1 (if model expects -1..1)
        # run inpaint at this scale
        inpainted = inpaint_one_scale(model, img_input, mask_s, steps=steps_per_scale, device=device)
        # convert back to 0..1
        inpainted = (inpainted + 1.0) / 2.0
        # save intermediate output
        save_tensor_as_image(inpainted, os.path.join(out_dir, f"inpaint_scale_{s}.png"))
        prev_inpaint = inpainted

    print("Progressive inpainting complete. Outputs saved to", out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to input image")
    parser.add_argument("--mask", required=True, help="path to binary mask (white=hole)")
    parser.add_argument("--out", default="./outputs", help="output dir")
    parser.add_argument("--ckpt", default="", help="path to mdm checkpoint (optional)")
    parser.add_argument("--scales", default="128,256,512", help="comma list of scales (coarse->fine)")
    parser.add_argument("--steps", type=int, default=50, help="sampling steps per scale")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scales = [int(x) for x in args.scales.split(",")]
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    # load model (replace with actual loader)
    model = load_mdm_model(args.ckpt, device)
    progressive_inpaint(model, args.image, args.mask, args.out, scales=scales, steps_per_scale=args.steps, device=device)

from PIL import Image
import torch
import torchvision.transforms.functional as TF
import numpy as np

def load_image_as_tensor(path):
    img = Image.open(path).convert("RGB")
    img = TF.to_tensor(img).float()  # [C,H,W], 0..1
    return img

def load_mask_as_tensor(path):
    m = Image.open(path).convert("L")  # single channel
    m = TF.to_tensor(m).float()
    # threshold to binary: assume white (255) is hole
    m = (m > 0.5).float()
    return m

def save_tensor_as_image(tensor, out_path):
    # tensor: [C,H,W] in 0..1
    t = tensor.clamp(0,1)
    img = TF.to_pil_image(t)
    img.save(out_path)

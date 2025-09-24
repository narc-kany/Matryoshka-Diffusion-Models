Here’s a polished, detailed `README.md` you can use (or adapt) for **narc-kany/Matryoshka-Diffusion-Models**. This version includes project overview, architecture, usage examples, installation, and contribution guidance. You can paste this into the repository’s README.

---

````markdown
# Matryoshka Diffusion Models (MDM)

> *“Unfolding creativity, scale by scale.”*

Matryoshka Diffusion Models (MDM) are a unified, nested diffusion framework that jointly handles multiple resolutions within a single model. Inspired by the Russian nesting dolls, MDM captures both global structure and fine detail simultaneously, enabling high-fidelity image generation, editing, and inpainting — all in one cohesive architecture.

---

## Table of Contents

1. [Motivation & Key Ideas](#motivation--key-ideas)  
2. [Architecture Overview](#architecture-overview)  
3. [Highlights & Advantages](#highlights--advantages)  
4. [Usage & Examples](#usage--examples)  
   - Installation  
   - Inference / Sampling  
   - Progressive Multi-Scale Inpainting Demo  
5. [Project Structure](#project-structure)  
6. [Tips & Best Practices](#tips--best-practices)  
7. [Future Directions](#future-directions)  
8. [Contributing & License](#contributing--license)

---

## Motivation & Key Ideas

Diffusion models have become central to generative AI, but scaling them to high resolution often requires cascaded models or latent-space techniques, which introduce complexity, inconsistency, or reliance on auxiliary networks.  

MDM’s core idea is to **nest multiple resolutions into a single network**. Rather than stacking separate models for low-resolution base generation and high-resolution refiners, MDM **shares features and representations across scales**, training a unified model that consistently synthesizes images from coarse layout to fine detail.

Key motivations include:

- **Efficiency**: Avoid the overhead of multiple models and stage transitions.  
- **Consistency**: Maintain semantic and spatial alignment across scales.  
- **Flexibility**: Support tasks like image generation, editing, and multi-scale inpainting within one framework.

---

## Architecture Overview

MDM combines several architectural and training strategies to realize multi-scale learning. Here’s how the different pieces fit together:

1. **Nested UNet**  
   A variant of the UNet architecture that processes multi-scale features simultaneously. It shares internal features across scales, allowing the network to reason about coarse structure and fine detail in a unified pass.

2. **Multi-Scale Latents**  
   Internally, the model represents the data at multiple resolutions (e.g., 128×128, 256×256, 512×512) during diffusion steps. These latent levels co-evolve and influence each other, so that high-resolution output is aware of lower-resolution context.

3. **Progressive Training**  
   Training begins on a lower resolution “base” and progressively expands to higher resolutions. This curriculum eases optimization, reduces early training costs, and helps the model gradually adapt to fine-scale structure.

4. **Multi-Scale Loss Functions**  
   Losses are applied at multiple scales, ensuring that each resolution’s output is consistent and aligned with both coarse semantics and fine texture. This helps prevent inconsistencies between levels.

5. **Joint Sampling (One Diffusion Process)**  
   At inference time, MDM executes a joint diffusion process across all scales, producing outputs in multiple resolutions from a single sampling run. This eliminates the need for cascading separate models.

---

## Highlights & Advantages

- **Single Model, Multi-Scale** — instead of multiple models, everything is handled in one architecture.  
- **Strong Coherence** — less risk of stage mismatch; outputs at each scale remain consistent.  
- **Pixel-Space Capability** — MDM works directly in pixel space (not only latent space), enabling direct high-res synthesis.  
- **Editing & Inpainting Support** — naturally suited for masked image editing across scales (coarse fill → refinement → fine detail).  
- **Generative Flexibility** — supports unconditional generation, conditional sampling (e.g. class or text prompts), and even video when extended spatio-temporally.

---

## Usage & Examples

### Installation

```bash
git clone https://github.com/narc-kany/Matryoshka-Diffusion-Models.git
cd Matryoshka-Diffusion-Models
pip install -r requirements.txt
````

> Note: If you integrate or depend on Apple’s official **ml-mdm** library or pretrained checkpoints, follow their README for model weights, license, and data setup.

### Inference / Sampling

You can generate images (or videos, if supported) with a command like:

```bash
python sample.py \
  --checkpoint path/to/mdm_checkpoint.pt \
  --resolution 512 \
  --prompt "A serene mountain landscape at sunset" \
  --num_samples 4 \
  --guidance_scale 7.5
```

*This is illustrative; your actual sampling script arguments may differ based on your implementation.*

### Progressive Multi-Scale Inpainting Demo

This repository includes (or you can add) a demo script that illustrates **progressive inpainting across scales**:

1. **Input image** (`input.jpg`) and **binary mask** (`mask.png`, white = region to fill).
2. Start at a small resolution (e.g. 128×128), inpaint masked region.
3. Upsample result, feed into next scale (256×256), inpaint further.
4. Final inpainting at the highest resolution (e.g. 512×512).

This scale-aware editing produces coherent, high-quality fills avoiding blur or artifacts.

---

## Project Structure

Here’s a suggested / typical layout for this kind of project:

```
.
├── data/                  # Datasets, masks, sample images
├── models/                # Model definitions (nested UNet, etc.)
├── scripts/               # Training, sampling, inpainting scripts
├── utils/                 # Helpers (data loading, image IO, masks)
├── checkpoints/           # Saved model weights
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

You may also include subfolders for experiments, logs, visualizations, etc.

---

## Tips & Best Practices

* **Start with low resolution** — use progressive training to stabilize early learning.
* **Warm-start inpainting** — use upsampled previous-scale results as initialization when moving to higher resolution.
* **Mask blending** — soften mask boundaries (feathering) to avoid harsh edges in output.
* **Sampling steps** — more steps give better fidelity but cost more compute; 20–100 steps per scale is a good tradeoff.
* **Use EMA weights** (exponential moving average of model weights) for sampling, which often yields better visual quality.
* **Checkpoint and save samples** routinely so you can monitor model behavior across scales.

---

## Future Directions

* **Spatio-temporal MDM** — extend nested architecture to video (time + space).
* **3D & volumetric generation** — nest scales in depth/volume dimensions.
* **Multimodal editing** — combine text, masks, style, segmentation maps for guided editing.
* **Domain adaptation** — use nested transfer learning for new domains or higher resolutions.
* **Interactive tools** — integrate MDM into image editors for real-time multi-scale edits.

---

## Contributing & License

Contributions are welcome! Feel free to open issues or submit pull requests. Some suggestions:

* Add new sampling strategies or optimizers
* Provide pretrained models / checkpoints
* Support additional tasks (e.g. video, 3D)
* Improve documentation, tutorials, and visual examples

Please include tests or notebooks demonstrating your changes.

**License:**
Specify your license here (e.g. MIT, Apache 2.0). If you incorporate code or weights from Apple’s ml-mdm or other sources, ensure license compatibility and attribution.

---

Thank you for checking out **Matryoshka Diffusion Models**! 🎨
Generate boldly, refine across scales, and help unfold the next generation of creative AI.

```

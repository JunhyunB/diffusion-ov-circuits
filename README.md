# Mechanistic Dissection of Cross-Attention Subspaces in Text-to-Image Diffusion Models

Official implementation of **"Mechanistic Dissection of Cross-Attention Subspaces in Text-to-Image Diffusion Models"** (AAAI 2026).

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://github.com/JunhyunB/diffusion-ov-circuits)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains code for analyzing and manipulating semantic concepts in text-to-image diffusion models through spectral decomposition of cross-attention layers. Our method reveals how concepts are encoded in low-dimensional spectral subspaces of OV (Output-Value) circuits.

<p align="center">
  <img src="figures/fig3.png" alt="Concept Isolation Results" width="100%">
</p>

*Spectral isolation of semantic concepts. For each concept, we show generated images with concept components removed (top) and with only concept components activated (bottom), using different percentages of spectral components.*

### Key Features

- **Spectral Analysis**: SVD-based decomposition of cross-attention OV matrices
- **Concept Localization**: Identify which singular vectors encode specific semantic concepts
- **Concept Manipulation**: Remove or amplify concepts by modulating spectral components
- **Progressive Visualization**: Visualize the effect of removing different percentages of concept-related singular vectors

### Setup

```bash
# Clone the repository
git clone https://github.com/JunhyunB/diffusion-ov-circuits.git
cd diffusion-ov-circuits

# Install dependencies
pip install torch torchvision
pip install diffusers transformers accelerate
pip install numpy matplotlib seaborn tqdm pillow
```

**Hugging Face Authentication:**

To use Stable Diffusion 2.1, you need to:
1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Accept the license for [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
3. Login via CLI:
```bash
huggingface-cli login
```

## Usage

### Quick Start

The repository provides example notebooks for different concepts:

1. **NSFW Content Removal** (`test_naked.ipynb`)
2. **Artistic Style Manipulation** (`test_picasso.ipynb`)

Simply open and run the notebooks to see the spectral analysis and concept manipulation in action.

### Basic Usage

The `ProgressiveSVController` class (defined in the notebooks) provides the main functionality:

```python
# Initialize controller (ProgressiveSVController class is defined in the notebooks)
controller = ProgressiveSVController()

# Define prompt pairs (base prompt, concept prompt)
prompt_pairs = [
    ("A portrait of a woman", "A portrait in Picasso's style"),
    # and so on...
]

# Analyze which singular vectors encode the concept
sv_infos = controller.analyze_sv_contributions(
    prompt_pairs,
    concept_name='picasso'
)

# Select top 20% of concept-related singular vectors
selected_svs = controller.select_top_svs(sv_infos, percentage=20)

# Remove the concept (set multiplier=0.0)
controller.create_sv_control_hooks(selected_svs, multiplier=0.0)

# Generate image without the concept
image = controller.generate_image("A cubist painting by Picasso")
```

### Progressive Visualization

Generate a visualization showing the effect of removing different percentages of singular vectors:

```python
generate_progressive_visualization(
    controller=controller,
    test_prompt="A cubist painting by Pablo Picasso",
    sv_infos=sv_infos,
    percentages=[10, 15, 20, 25, 30],
    multiplier=0.0,  # 0.0 for removal, >1.0 for amplification
    save_path="picasso_removal.png"
)
```

## Method Overview

Our approach involves three main steps:

1. **SVD Decomposition**: Decompose each cross-attention head's OV matrix into singular vectors
2. **Contribution Analysis**: Measure how much each singular vector contributes to the concept by comparing base and concept prompt embeddings
3. **Spectral Modulation**: Selectively nullify or amplify concept-related singular vectors during generation

<p align="center">
  <img src="figures/fig2.png" alt="Spectral vs Head-level Modulation" width="90%">
</p>

*Comparison between spectral-level and head-level modulation. (Top) Scaling identified spectral components versus (bottom) scaling entire attention head outputs. The current code implements spectral-level modulation.*

For more details, please refer to our [paper](https://github.com/JunhyunB/diffusion-ov-circuits) (currently not available).


## License

This project is released under the MIT License.


## Contact

For questions or issues, please open an issue on GitHub or contact:
- Jun-Hyun Bae: junhyun.bae.kr@gmail.com
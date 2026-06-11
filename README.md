# Inkdiffuser

Inkdiffuser is a PyTorch-based font generation and diffusion project focused on ink spread, stroke structure, and image quality evaluation.

## Highlights

- Training pipeline built on `accelerate` and `diffusers`
- Content, style, and high-frequency feature encoders
- Differentiable ink-structure losses for diffusion-aware supervision
- Utility scripts for rendering TTF fonts, composing samples, and evaluation

## Repository Layout

- `train.py`: training entry point
- `eval.py`: image quality evaluation metrics
- `ttf_to_img.py`: render TTF characters into images
- `src/`: model and loss implementations
- `configs/`: argument parsers and configuration helpers
- `dataset/`: dataset and dataloader utilities
- `utils/`: normalization, checkpoint, and helper functions

## Installation

Install the dependencies first:

```bash
pip install -r requirements.txt
```

## Training

Run training with the configuration defined in `configs/fontdiffuser.py`:

```bash
python train.py
```

## Evaluation

Evaluate generated images against ground truth folders:

```bash
python eval.py --input_images_path <gt_dir> --generated_images_path <gen_dir>
```

## TTF to Image

Render characters from a TTF file:

```bash
python ttf_to_img.py --font_style <font.ttf> --chara <chars.txt> --save_path <output_dir>
```

## Notes

- Some utility scripts still contain local absolute paths in their default arguments.
- Large generated artifacts, checkpoints, and result folders are ignored by `.gitignore`.

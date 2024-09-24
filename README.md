# Image Colorization on Urban Landscapes

## Overview

This project focuses on the task of **image colorization** — predicting RGB colors for grayscale images — using various **CNN-based architectures**. The primary goal is to compare traditional CNN models against advanced pre-trained models, such as **Vision Transformers (ViT)**, to improve the colorization quality of **urban landscapes** extracted from a subset of the **SUN dataset**. The models are evaluated using metrics like **PSNR**, **LPIPS**, and **SSIM**.

## Dataset

We used a manually selected subset of the **SUN dataset** — a large collection of images for scene understanding tasks — to focus on urban landscape images. From the **SUN397** dataset, we selected **32 categories** (e.g., streets, buildings, parks), totaling **12,950 images**. The data was pre-processed and resized to 256x256 before being input to the model.

### Preprocessing

1. Images are resized to 256x256.
2. Converted from **RGB** to **CIELAB color space**, where only the **L channel** (lightness) are used for input.
3. Normalized all channels to a range of [-1, 1].

## Methods

The project employs **CNN-based encoder-decoder architectures** for the colorization task, with some models incorporating pre-trained feature extractors like **Inception-V3** and **Vision Transformers (ViT)**.

### CNN Encoder-Decoder Model

The CNN model uses an **encoder** to extract features from the grayscale image and a **decoder** to reconstruct the image with color. We trained multiple models:
- A **baseline CNN** (Hourglass model) with 3 convolutional layers.
- **Inception-V3**: Pre-trained on ImageNet, used for feature extraction.
- **Vision Transformer (ViT)**: Pre-trained transformer model, capturing global image features.

### Model Architectures

1. **Baseline CNN**: Simple encoder-decoder architecture with no pre-trained feature extractor.
2. **CNN + Inception-V3**: Combines CNN with Inception-V3 as a feature extractor.
3. **CNN + Vision Transformer (ViT)**: Combines CNN with ViT for feature extraction.

## Experiments

### Training

- **Optimizer**: Adam with a learning rate of 0.0001, weight decay of 1e-6.
- **Batch size**: 64
- **Epochs**: 50
- **Hardware**: 1 NVIDIA T4 GPU.

Preprocessing was performed in advance to reduce computational overhead.

### Hyperparameters

| Hyperparameter | Value     |
|----------------|-----------|
| Epochs         | 50        |
| Learning Rate  | 1e-4      |
| Batch Size     | 64        |
| Weight Decay   | 1e-6      |

## Results

We evaluated the models using the following metrics:

- **PSNR (Peak Signal Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**

### Performance Metrics

| Model                 | PSNR  | SSIM   | LPIPS | MSE   |
|-----------------------|-------|--------|-------|-------|
| CNN Baseline           | 22.05 | 0.8712 | 0.1725 | 0.0091 |
| CNN + Inception-V3     | 25.10 | 0.9390 | 0.1158 | 0.0073 |
| CNN + Vision Transformer | 25.00 | 0.9375 | **0.1106** | 0.0076 |

The ViT model performed better on the **LPIPS** metric, indicating a higher perceptual similarity in colorization compared to other models.

### Best Performing Categories

- **Basilica**: PSNR = 27.42, SSIM = 0.9555
- **Skyscraper**: PSNR = 25.68, SSIM = 0.9468

## Failure Cases

The model struggled with complex urban scenes or images with multiple objects, and there was a tendency to over-predict green colors, especially in vegetation. These cases were particularly evident when there were patterns or cluttered objects in the image.

## Testing on Historical Urban Images

The model was also tested on historical images from **Padova**, Italy. It performed well on simpler structures like **Porta Portello**, but struggled with complex scenes like **Pratto della Valle**, where it failed to colorize the concrete areas effectively.

## Conclusion

Using CNN encoder-decoder architectures was effective for image colorization. The addition of pre-trained models (Inception-V3 and ViT) helped enhance performance, particularly in **LPIPS** and **perceptual similarity**. However, the model faced challenges in complex scenes, and there is room for improvement in color generalization across different object types.

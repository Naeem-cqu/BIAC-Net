# BIAC-Net: Bidirectional Global-Local Communication for Feature Refinement in Medical Image Classification

Official implementation of BIAC-Net, a hybrid medical image classification framework that combines a Vision Transformer backbone with global-local dual-branch refinement, bidirectional feature communication, and adaptive gated fusion.

## Overview

Medical image classification often requires both:

- global contextual semantics** for broader anatomical understanding
- fine-grained local structural cues** for subtle lesion patterns

BIAC-Net addresses this by combining:

- a ViT-B/16** backbone for contextual feature extraction
- a CBAM-based global refinement branch**
- a depthwise separable convolution-based local refinement branch**
- a idirectional cross-stream residual communication module**
- an adaptive gated fusion module**

The framework is evaluated on Kvasir and ISIC 2018 for medical image classification.

## Main Features

- Vision Transformer backbone with end-to-end fine-tuning
- Dual-branch global-local feature refinement
- Bidirectional communication between global and local streams
- Adaptive gated fusion for input-dependent feature integration
- Grad-CAM visualization for qualitative interpretability
- Support for Kvasir and ISIC 2018 experiments

## Method Summary

Given a ViT feature tensor, BIAC-Net constructs two complementary branches:

- Global branch: enhances contextual saliency using CBAM
- Local branch: preserves fine-grained structural patterns using depthwise separable convolution

The two streams are then updated through iterative bidirectional communication:

- global features receive projected local information
- local features receive projected global information

After communication, the refined features are fused using an adaptive gate and passed to the classifier.

## Datasets

This repository supports experiments on:

### 1. Kvasir
A gastrointestinal endoscopy image dataset with 8 classes and 4,000 images.

### 2. ISIC 2018
A dermoscopic skin lesion classification dataset with 7 classes and 10,015 images.

### Data Split
The experiments use a 70% / 10% / 20% split for training, validation, and testing.

## Environment

Recommended setup:

- Python 3.9+
- PyTorch
- Torchvision
- NumPy
- OpenCV
- scikit-learn
- matplotlib
- Pillow
- tqdm

##You can install the main dependencies with:

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib pillow tqdm


# Vision Transformer on CIFAR-10

This project implements a **Vision Transformer (ViT)** from scratch in PyTorch and trains it on the **CIFAR-10** image classification dataset.

## Overview
The model is built using the core components of a Vision Transformer:
- **Patch Embedding** with Conv2D  
- **Learnable [CLS] token** and **positional embeddings**  
- **Transformer encoder blocks** with Multi-Head Self-Attention and MLP layers  
- **Residual connections** and **LayerNorm**  
- A linear head for classification into 10 CIFAR-10 classes  

The notebook also includes:
- Data loading and preprocessing (normalization + augmentations)  
- Training and evaluation loops  
- Accuracy tracking and plotting  

## Environment
- Implemented and trained on **Google Colab**  
- Hardware: **T4 GPU** runtime  

## Usage
Open the Jupyter notebook on Colab and run all cells. The dataset will be downloaded automatically, and the model will begin training.  

## Notes
This is a simple, educational implementation to understand how Vision Transformers work.  
On CIFAR-10, CNNs often outperform ViTs unless stronger augmentation and regularization are added, but this notebook provides a solid starting point for experimentation.

# Autoencoders

## Overview

This project comprises three Python scripts, each implementing different types of autoencoders using Keras. Autoencoders are neural networks used for unsupervised learning of efficient codings. The three types of autoencoders implemented are:

1. **Vanilla Autoencoder** (`0-vanilla.py`)
2. **Sparse Autoencoder** (`1-sparse.py`)
3. **Convolutional Autoencoder** (`2-convolutional.py`)

Each script is designed to handle different types of data and provide various benefits in terms of performance and output quality.

## File Descriptions

### `0-vanilla.py`

**Purpose:**
This script implements a Vanilla Autoencoder. Vanilla Autoencoders are the simplest form of autoencoders and consist of only fully connected layers.

**Components:**

1. **Encoder:**
   - Uses fully connected layers to compress the input data into a lower-dimensional latent representation.

2. **Decoder:**
   - Uses fully connected layers to reconstruct the input data from the latent representation.

3. **Autoencoder:**
   - Combines the encoder and decoder into one model.

**Why We Did It:**
Vanilla Autoencoders are a good starting point for understanding the basic structure and functionality of autoencoders. They are easy to implement and can be used on simple datasets for dimensionality reduction or noise reduction.

### `1-sparse.py`

**Purpose:**
This script implements a Sparse Autoencoder. Sparse Autoencoders introduce sparsity constraints on the latent representations, forcing the network to learn more efficient and useful features.

**Components:**

1. **Encoder:**
   - Uses fully connected layers with L1 regularization to enforce sparsity on the latent representation.

2. **Decoder:**
   - Uses fully connected layers to reconstruct the input data from the sparse latent representation.

3. **Autoencoder:**
   - Combines the encoder and decoder into one model with a sparsity constraint.

**Why We Did It:**
Sparse Autoencoders are useful for learning more efficient representations of the input data by enforcing sparsity. This can lead to better feature extraction and improved performance on tasks such as anomaly detection or image reconstruction.

### `2-convolutional.py`

**Purpose:**
This script implements a Convolutional Autoencoder. Convolutional Autoencoders are designed to work with image data and use convolutional layers to exploit the spatial structure of the data.

**Components:**

1. **Encoder:**
   - Uses convolutional layers to compress the input image data into a lower-dimensional latent representation while preserving spatial hierarchies.

2. **Decoder:**
   - Uses transposed convolutional layers to reconstruct the input image data from the latent representation.

3. **Autoencoder:**
   - Combines the encoder and decoder into one model tailored for image data.

**Why We Did It:**
Convolutional Autoencoders are particularly effective for image data due to their ability to capture spatial hierarchies and patterns. They are widely used in applications such as image denoising, inpainting, and compression.

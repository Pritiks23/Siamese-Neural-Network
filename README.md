


# Siamese Image Similarity Neural Network

Check me out here: https://pritiks23.github.io/Siamese-Neural-Network/

This repository demonstrates a **Siamese neural network** for image similarity, fully implemented in-browser using **TensorFlow.js**. MobileNet serves as the shared backbone, converting images into high-dimensional embeddings for pairwise comparison.

## Architecture Overview
<img width="700" height="300" alt="Screen Shot 2025-09-05 at 9 56 01 AM" src="https://github.com/user-attachments/assets/ee1bb62c-2d3e-4ff3-89bb-cb3e1191726e" />
1. **Shared Backbone (MobileNet v2.1)**  
   - Pre-trained convolutional feature extractor.
   - Outputs feature maps from the final convolutional layers.
   - Embeddings are generated via global average pooling on feature maps.
   - Optional **L2-normalization** ensures embedding vectors reside on a hypersphere for stable cosine similarity.

2. **Pairwise Comparison**  
   - **Cosine similarity**: \( \text{sim}(a,b) = \frac{a \cdot b}{\|a\| \|b\|} \)
   - **Euclidean distance**: \( d(a,b) = \sqrt{\sum_i (a_i - b_i)^2} \)
   - Embeddings can be used directly or fed into a trainable head.

3. **Trainable Head (Few-shot Classifier)**  
   - Input: element-wise absolute difference of embeddings, \( |e_a - e_b| \)  
   - Layers: Dense(256, ReLU) → Dropout(0.25) → Dense(64, ReLU) → Dense(1, Sigmoid)  
   - Optimizer: Adam; Loss: Binary Cross-Entropy  
   - Outputs a similarity probability between 0 and 1.  

4. **Caching & Memory Management**  
   - Embeddings cached to prevent redundant computations.
   - Tensors disposed after inference to minimize GPU/CPU memory footprint.


## Usage

1. Upload or drag-and-drop two images.
2. Compare embeddings:
   - Cosine similarity and Euclidean distance are computed from backbone embeddings.
3. Optionally:
   - Add labeled pairs (similar/different).
   - Train the head model in-browser.
   - Inspect predictions on new pairs.
4. Download trained head model for reuse or deployment.

## Technical Highlights

- **End-to-end in-browser inference**: no server required.  
- **Few-shot training**: supports small datasets (10–100 pairs).  
- **Embedding normalization**: stabilizes gradient flow in head training.  
- **Modular architecture**: backbone embeddings decoupled from classifier head, allowing easy replacement of the backbone or head.  

## Deployment

- Static HTML/JS; compatible with GitHub Pages.  
- Dependencies loaded via CDN: `@tensorflow/tfjs`, `@tensorflow-models/mobilenet`.

## References

- [TensorFlow.js](https://www.tensorflow.org/js)  
- [MobileNet: Efficient CNN Architecture](https://arxiv.org/abs/1704.04861)  
- [Siamese Networks for One-Shot Learning](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

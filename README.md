# **CS 5780: Siamese Network for Relational Image Classification**

This repository contains my winning submission for the "Rock-Paper-Scissors" Kaggle competition, part of the CS 5780 Machine Learning course at Cornell University. The final model achieved **88% accuracy** on the private test set, surpassing all three official benchmarks.

### **Project Objective**
The goal was to build a machine learning model that takes two 24x24 grayscale images of hand gestures as input and predicts if the first image wins against the second (`+1`) or not (`-1`). The challenge required building a model that could understand the *relationship* between two images, not just their individual classes.

### **Model Architecture**
To tackle this relational problem, I implemented a **Siamese Neural Network**. This architecture uses two identical feature extractors to process each image and then compares the resulting feature vectors.

* **Feature Extractor**: The core of the network is a custom ResNet-style feature extractor built in PyTorch.
* **Residual Blocks**: The extractor uses Residual Blocks to enable deeper layers and improve gradient flow during training.
* **Squeeze-and-Excitation (SE) Blocks**: To refine features, SE blocks were integrated into the residual architecture. These blocks adaptively re-weigh channel-wise features, allowing the model to focus on the most informative parts of the images.

### **Training & Methodology**
The model was trained end-to-end with a focus on maximizing generalization to the private leaderboard.

* **Loss Function & Optimizer**: I used `nn.SoftMarginLoss()` with the `AdamW` optimizer.
* **Data Augmentation**: To prevent overfitting and create a more robust model, I applied a wide range of transformations to the training data, including random rotations, flips, scaling, and color jitter.
* **Learning Rate Scheduling**: A `ReduceLROnPlateau` scheduler automatically decreased the learning rate when validation accuracy stopped improving.
* **Early Stopping**: Training was halted automatically if validation accuracy did not improve for a set number of epochs, saving the best-performing model state.
* **Test-Time Augmentation (TTA)**: For final predictions, I used TTA. The model made predictions on both the original test images and their horizontally flipped versions. The final output was the average of these predictions, which provided a significant boost in accuracy.

### **Technologies Used**
* PyTorch
* Scikit-learn
* Pandas & NumPy

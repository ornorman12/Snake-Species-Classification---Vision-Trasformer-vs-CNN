# Snake-Species-Classification---Vision-Trasformer-vs-CNN


## Overview

This project aims to classify snake species using two deep learning models: **ResNet-50** and **Vision Transformer (ViT) - DINOv2**. The models are trained and evaluated on a custom dataset sourced from Kaggle, with the objective of comparing their performance in terms of accuracy and robustness, both with and without data augmentation techniques.

## Models

### 1. ResNet-50

**ResNet-50** is a deep convolutional neural network that is part of the Residual Network (ResNet) family. Introduced by He et al. in 2015, ResNet models address the vanishing gradient problem by incorporating residual connections or "skip connections," which allow gradients to flow through the network more effectively during training.

#### Key Features:
- **Residual Connections:** The hallmark of ResNet-50 is its use of residual blocks, which help the model maintain performance as the network depth increases. This is particularly useful for training very deep networks.

![Alt text](https://api.wandb.ai/files/mostafaibrahim17/images/projects/37042936/3a93c9fd.png)


### 2. Vision Transformer (ViT) - DINOv2

**Vision Transformer (ViT)** represents a shift from traditional convolutional neural networks to transformer-based architectures, originally designed for natural language processing. ViT applies the transformer model to image patches, treating them as sequences similar to words in a sentence.

#### Key Features:
- **Patch-Based Processing:** Instead of using convolutions, ViT divides images into patches and processes them as a sequence, allowing the model to capture long-range dependencies and global context.


![Alt text](https://miro.medium.com/v2/resize:fit:1358/1*inc9Sty8xMFNNYlNVn9iBQ.png)
### Comparison Between ResNet-50 and ViT - DINOv2

Both **ResNet-50** and **ViT - DINOv2** offer distinct advantages:

- **ResNet-50**: A more traditional model that excels in capturing fine-grained details through deep layers, particularly effective in scenarios where convolutional approaches have been successful.
  
- **ViT - DINOv2**: A modern transformer-based model that captures global image features, making it suitable for tasks that require understanding relationships across the entire image.

In this project, both models were trained and evaluated to determine which architecture is more effective for the task of snake species classification.

## Installation

To run this project, you'll need to set up a Python environment with the necessary dependencies. You can install the required packages using:

```bash
pip install -r requirements.txt
```


## Dataset

### Source

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/goelyash/165-different-snakes-species), a widely used platform for datasets. It originally contained a large and diverse collection of images representing various snake species.

### Dataset Filtering

After downloading the dataset, a rigorous filtering process was applied to ensure data quality and relevance. The steps included:

- **Class Balancing**: Classes with insufficient data were removed, thus preventing the model from being skewed towards certain species with more data.

### Final Dataset Structure

The final dataset, after filtering, was organized into training and test sets:

- **train/**: Contains the filtered and balanced training images, used for training the models.
- **test/**: Contains the filtered test images, used for evaluating the models' performance.
- **train_new.csv** and **test_new.csv**: These CSV files include the labels and metadata for the filtered images, providing necessary details for model training and evaluation.


## Results

The following results were obtained after training and evaluating the ResNet-50 and ViT - DINOv2 models. The performance was assessed using Top-1, Top-3, and Top-5 accuracy metrics, both with and without data augmentation.

### Without Augmentation

#### ResNet-50:
- **Top-1 Accuracy:** 65.14%
- **Top-3 Accuracy:** 83.32%
- **Top-5 Accuracy:** 88.79%

#### ViT - DINOv2:
- **Top-1 Accuracy:** 75.60%
- **Top-3 Accuracy:** 89.66%
- **Top-5 Accuracy:** 93.61%

### After Applying Augmentation

#### ResNet-50:
- **Top-1 Accuracy:** 65.95% (Increase of 0.81%)
- **Top-3 Accuracy:** 84.43% (Increase of 1.11%)
- **Top-5 Accuracy:** 89.42% (Increase of 0.63%)

#### ViT - DINOv2:
- **Top-1 Accuracy:** 76.35% (Increase of 0.75%)
- **Top-3 Accuracy:** 90.47% (Increase of 0.81%)
- **Top-5 Accuracy:** 93.72% (Increase of 0.11%)

### Analysis

- **ViT - DINOv2** consistently outperformed **ResNet-50** across all metrics. The Vision Transformer model showed a superior ability to correctly identify the true class as the top prediction (Top-1 accuracy) compared to ResNet-50.
  
- **Impact of Augmentation**:
 The application of data augmentation led to mild improvements across all accuracy metrics, with a slightly better improvement compared to ViT - DINOv2.

## Future Work

To further improve the performance and robustness of the models, the following areas of exploration are recommended:

### 1. Explore More Augmentation Techniques

- **Diversified Augmentation Strategies:** Further experimentation with different data augmentation techniques could help in improving model performance.
  
- **Augmentation Tuning:** Fine-tuning the intensity and probability of existing augmentations might also yield better results, ensuring that the augmentations provide beneficial variability without distorting the data too much.

### 2. Compare Larger Model Variants

- **Larger ResNet Variants:** Testing larger variants of ResNet, such as ResNet-101 or ResNet-152, could offer insights into whether deeper architectures provide better performance on this classification task. These models might also respond differently to data augmentation strategies.
  
- **Extended Vision Transformer Models:** Similarly, experimenting with larger Vision Transformer models that have more parameters or layers could potentially enhance performance, especially in handling complex image features.

### 3. Hyperparameter Tuning

- **Learning Rate and Batch Size:** Further tuning of hyperparameters such as learning rate, batch size, and optimizer configurations could lead to more optimal training conditions, resulting in better model performance.
    
By addressing these areas, it may be possible to unlock additional performance gains and further optimize the models for the task of snake species classification.





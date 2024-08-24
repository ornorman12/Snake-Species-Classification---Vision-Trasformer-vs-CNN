# Snake-Species-Classification---Vision-Trasformer-vs-CNN


## Overview

This project aims to classify snake species using two deep learning models: **ResNet-50** and **Vision Transformer (ViT) - DINOv2**. The models are trained and evaluated on a custom dataset sourced from Kaggle, with the objective of comparing their performance in terms of accuracy and robustness, both with and without data augmentation techniques.


## Installation

To run this project, you'll need to set up a Python environment with the necessary dependencies. You can install the required packages using:

```bash
pip install -r requirements.txt


## Dataset

### Source

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/), a widely used platform for datasets. It originally contained a large and diverse collection of images representing various snake species.

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
  - **ResNet-50**: The application of data augmentation led to mild improvements across all accuracy metrics, with a slightly better improvement compared to ViT - DINOv2. This suggests that ResNet-50 benefits more from augmentation, likely due to its architecture.
  - **ViT - DINOv2**: Although the improvements were also mild, they indicate that the ViT - DINOv2 model is robust and performs well even with minor enhancements from augmentation techniques.

Overall, **ViT - DINOv2** remains the superior model in this classification task, though both models demonstrated slight performance gains with the application of data augmentation.




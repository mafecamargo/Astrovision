# Disaster Image Classification using Deep Convolutional Neural Networks

## Overview

This project investigates the use of Deep Convolutional Neural Networks (CNNs) for automatic disaster image classification. The objective is to train and evaluate multiple state-of-the-art image classification architectures and compare their performance on a disaster-related image dataset.

The project follows a complete machine learning workflow:

1. Dataset preparation and preprocessing
2. Model training using transfer learning
3. Validation and testing
4. Quantitative model comparison
5. Interactive deployment using Streamlit

The final system allows users to upload an image and obtain a disaster category prediction together with confidence scores.

# Motivation

Natural disasters such as floods, earthquakes, wildfires, and landslides can cause significant damage to infrastructure, ecosystems, and human populations.

During emergency response operations, large quantities of images are collected from:

- Social media
- Surveillance cameras
- Drones
- Satellites
- Mobile devices

Manual inspection of these images is time-consuming and difficult to scale. Automatic image classification systems can support emergency management teams by rapidly categorizing incoming visual information and identifying affected regions.

This project explores whether modern CNN architectures can effectively distinguish between different disaster categories using transfer learning.

---

# Dataset

The dataset consists of images grouped into disaster-related categories.

```text
data/
├── Damaged_Infrastructure
│   ├── Earthquake
│   └── Infrastructure
│
├── Fire_Disaster
│   ├── Urban_Fire
│   └── Wild_Fire
│
├── Land_Disaster
│   ├── Drought
│   └── Land_Slide
│
├── Non_Damage
│   ├── Non_Damage_Buildings_Street
│   ├── Non_Damage_Wildlife_Forest
│   └── sea
│
└── Water_Disaster
```

The current implementation considers the first-level folders as classes:

| Class ID | Class Name             |
| -------- | ---------------------- |
| 0        | Damaged_Infrastructure |
| 1        | Fire_Disaster          |
| 2        | Land_Disaster          |
| 3        | Non_Damage             |
| 4        | Water_Disaster         |

Thus, the task is a 5-class image classification problem.

---

# Model Architectures

Two CNN architectures are evaluated.

## 1. VGG16

VGG16 is a deep convolutional neural network introduced by the Visual Geometry Group (VGG) at the University of Oxford.

Characteristics:

- 16 learnable layers
- Sequential architecture
- Uses small 3×3 convolution kernels
- Approximately 138 million parameters

Advantages:

- Simple architecture
- Strong baseline model
- Easy to understand

Limitations:

- Large memory footprint
- Slower training and inference
- Higher risk of overfitting

---

## 2. ResNet50

ResNet50 was introduced by Microsoft Research and introduced the concept of residual learning.

Characteristics:

- 50 layers
- Residual (skip) connections
- Approximately 25 million parameters

Advantages:

- Deeper network
- Better gradient propagation
- Faster convergence
- Generally better accuracy

Limitations:

- More complex architecture
- Less interpretable than VGG

# Evaluation Metrics

Models are compared using accuracy, precision, recall, F1 Score and a Confusion Matrix.

---

# Training a Model

Example using ResNet50:

```bash
python train.py
```

Inside `train.py`:

```python
MODEL_NAME = "resnet50"
```

For VGG16:

```python
MODEL_NAME = "vgg16"
```

The best model checkpoint will be saved automatically.

---

# Streamlit Application

The project includes an interactive deployment interface.

Launch:

```bash
streamlit run app.py
```

The web application allows users to:

1. Upload an image
2. Run inference using the trained model
3. View predicted class
4. View confidence score
5. Visualize class probabilities

---

# Expected Workflow

Train ResNet50:

```bash
python train.py
```

Launch application:

```bash
streamlit run app.py
```

Upload an image:

```text
flood_scene.jpg
```

Example output:

```text
Predicted Class: Water_Disaster

Confidence: 94.7%
```

---

# Future Improvements

Potential extensions include:

- EfficientNet
- DenseNet
- Vision Transformers (ViT)
- Hyperparameter optimization
- K-fold cross-validation
- Explainability methods (Grad-CAM)
- Multi-label disaster classification
- Deployment using Docker
- REST API integration

---

# References

- Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- PyTorch Documentation: https://pytorch.org

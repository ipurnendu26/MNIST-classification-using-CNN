# MNIST Handwritten Digit Classification using CNN

A deep learning project that classifies handwritten digits (0-9) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deep Learning Lifecycle](#deep-learning-lifecycle)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project demonstrates a complete deep learning workflow for image classification using the MNIST dataset. The CNN model achieves **~99% accuracy** on the test set.

### Key Features
- Complete data science lifecycle implementation
- CNN with BatchNormalization and Dropout for regularization
- Training callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Comprehensive model evaluation and error analysis
- Model saving and inference demo

## 📊 Dataset

**MNIST** (Modified National Institute of Standards and Technology) dataset:
- **Total Images:** 70,000 grayscale images
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Image Size:** 28×28 pixels
- **Classes:** 10 (digits 0-9)

The dataset is automatically downloaded via `tensorflow.keras.datasets`.

## 📁 Project Structure

```
MNIST-classification-using-CNN/
│
├── MNIST Deep learning.ipynb    # Main Jupyter notebook
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore file
│
├── venv/                        # Virtual environment (not tracked)
│
└── Generated Files (after training):
    ├── mnist_cnn_final.keras    # Saved model
    ├── best_model.keras         # Best checkpoint
    ├── mnist_cnn_weights.weights.h5
    ├── training_history.json
    ├── training_history.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    ├── misclassified_samples.png
    └── model_architecture.png
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MNIST-classification-using-CNN.git
   cd MNIST-classification-using-CNN
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install Graphviz for model visualization**
   ```bash
   # Windows (using winget)
   winget install Graphviz.Graphviz

   # Linux
   sudo apt-get install graphviz

   # Mac
   brew install graphviz
   ```

## 💻 Usage

### Running the Notebook

1. Open VS Code or Jupyter Lab
2. Open `MNIST Deep learning.ipynb`
3. Select the `venv` kernel
4. Run all cells sequentially

### Quick Inference

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_final.keras')

# Predict a digit (image should be 28x28 grayscale)
def predict_digit(model, image):
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
    prediction = model.predict(image, verbose=0)
    return np.argmax(prediction), prediction[0].max()

# Example usage
digit, confidence = predict_digit(model, your_image)
print(f"Predicted: {digit}, Confidence: {confidence:.2%}")
```

## 🏗️ Model Architecture

```
Model: "MNIST_CNN"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv1 (Conv2D)              (None, 28, 28, 32)        320       
bn1 (BatchNormalization)    (None, 28, 28, 32)        128       
pool1 (MaxPooling2D)        (None, 14, 14, 32)        0         
conv2 (Conv2D)              (None, 14, 14, 64)        18,496    
bn2 (BatchNormalization)    (None, 14, 14, 64)        256       
pool2 (MaxPooling2D)        (None, 7, 7, 64)          0         
conv3 (Conv2D)              (None, 7, 7, 64)          36,928    
bn3 (BatchNormalization)    (None, 7, 7, 64)          256       
flatten (Flatten)           (None, 3136)              0         
dense1 (Dense)              (None, 128)               401,536   
dropout (Dropout)           (None, 128)               0         
output (Dense)              (None, 10)                1,290     
=================================================================
Total params: 459,210
Trainable params: 458,890
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 64 |
| Epochs | 20 (with EarlyStopping) |
| Validation Split | 20% |

## 📈 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~99.2% |
| **Test Loss** | ~0.03 |
| **Training Time** | ~5 minutes (CPU) |

### Per-Class Performance
All digit classes achieve >98% accuracy, with some confusion between similar digits (e.g., 4↔9, 3↔5).

## 🔄 Deep Learning Lifecycle

This project follows a complete deep learning workflow:

1. ✅ **Problem Definition** - Multi-class image classification
2. ✅ **Data Collection** - MNIST dataset (70,000 images)
3. ✅ **EDA** - Class distribution, sample visualization
4. ✅ **Data Preprocessing** - Normalization, reshaping, one-hot encoding
5. ✅ **Model Design** - CNN with BatchNorm, Dropout
6. ✅ **Training** - With callbacks for optimization
7. ✅ **Evaluation** - Accuracy, confusion matrix, classification report
8. ✅ **Error Analysis** - Misclassification patterns
9. ✅ **Deployment** - Model saving and inference

## 🔮 Future Improvements

- [ ] Data Augmentation (rotation, zoom, shift)
- [ ] Hyperparameter tuning with Keras Tuner
- [ ] Try deeper architectures (ResNet, VGG-like)
- [ ] Ensemble methods
- [ ] Model quantization for edge deployment
- [ ] Web interface using Streamlit/Gradio
- [ ] ONNX export for cross-platform inference

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) - Yann LeCun
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning framework
- [Scikit-learn](https://scikit-learn.org/) - ML utilities

---

⭐ If you found this project helpful, please give it a star!

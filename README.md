# 🧠 Brain Tumor Detection Using CNN

This project leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to detect brain tumors from MRI scans. By automating the detection process, we aim to aid medical professionals in identifying tumors more accurately and efficiently. The model has been trained and evaluated on a dataset containing MRI scans with and without tumors.

## 📝 Project Overview

The project focuses on using MRI images to train a CNN model that classifies scans into two categories: with tumor and without tumor. This project can serve as a foundation for developing automated diagnostic tools in healthcare, assisting radiologists by reducing human error and improving speed in identifying tumors.

### Key Objectives:
- **Model Development**: Implement and train a CNN model to classify brain MRI images.
- **Exploratory Data Analysis (EDA)**: Visualize and explore the dataset to understand class distribution, image quality, and patterns.
- **Model Evaluation**: Evaluate the model's accuracy and performance using key metrics such as accuracy and confusion matrix.

## 📊 Data Summary

- **Source**: Brain tumor MRI image dataset from [your dataset source].
- **Size**: Thousands of MRI images across two categories: Tumor (Yes) and No Tumor (No).
  
### Key Columns:
- **Image Data**: MRI scan images (grayscale or RGB format).
- **Tumor Presence**: Binary classification (1: Tumor, 0: No Tumor).

## 🧑‍💻 Methodology

### Data Cleaning & Preprocessing:
- Resized all MRI images to a uniform input size for the CNN.
- Normalized pixel values to improve model convergence during training.
- Split dataset into training and test sets using an 80-20 ratio.

### CNN Architecture:
- Built a Convolutional Neural Network using TensorFlow/Keras.
- Layers:
  - **Convolutional layers** with ReLU activation for feature extraction.
  - **Pooling layers** for dimensionality reduction.
  - **Fully connected layers** for classification.
  - **Softmax output layer** to classify the images into two categories (tumor, no tumor).

### Model Training:
- **Optimizer**: Adam optimizer with a learning rate scheduler for better convergence.
- **Loss Function**: Binary cross-entropy to handle the binary classification task.
- **Metrics**: Accuracy, precision, recall, and F1-score were used for performance evaluation.

## 📈 Results

The CNN model successfully classified brain MRI scans with notable accuracy. Below are key takeaways from the model's performance:

### Evaluation Metrics:
- **Accuracy**: Achieved over 90% accuracy on the test set.
- **Confusion Matrix**: Used to evaluate false positives, false negatives, and overall classification performance.
  
### Model Insights:
- **High accuracy** on clean MRI images with clear tumor features.
- **Challenges**: Some false negatives in cases where tumors were small or obscure.

## 📊 Visualization and Insights

- **Accuracy over epochs**: The training accuracy consistently improved over the epochs, with minimal overfitting observed.
- **Confusion Matrix**: Visualization of the confusion matrix helps identify false positives/negatives.
- **Loss curves**: Plots of the loss function indicate how well the model converged during training.

## 🛠️ Recommendations for Improvement

- **Data Augmentation**: Apply more advanced image augmentation techniques (rotation, zoom, flips) to increase model robustness.
- **Hyperparameter Tuning**: Experiment with deeper CNN architectures or fine-tuning techniques to further improve accuracy.
- **Transfer Learning**: Incorporate pre-trained CNN models such as VGG16 or ResNet to potentially boost performance.

## 🚀 Future Work

- **Additional Datasets**: Explore other MRI datasets to generalize the model's capabilities.
- **Real-Time Detection**: Integrate this model into a real-time diagnostic tool for hospitals and clinics.
- **Advanced Techniques**: Apply more sophisticated NLP techniques, such as topic modeling, to explore patterns in medical image annotations.

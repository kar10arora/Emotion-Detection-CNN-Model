# Emotion Detection Model

## Overview

This project implements an emotion detection model using a stacked CNN architecture. The model is trained on the FER2013 dataset and achieves an accuracy of approximately 67% after extensive data augmentation. Due to system limitations, further training was not possible.

# Features

Utilizes a deep convolutional neural network (CNN) for emotion recognition.

Implements data augmentation techniques to improve generalization.

Uses TensorFlow and Keras for model development.

Incorporates early stopping and model checkpointing for better training efficiency.

Evaluates the model using accuracy and loss metrics.

# Dataset

The model is trained on the FER2013 dataset, downloaded from Kaggle.

Contains grayscale images categorized into various emotions such as happy, sad, angry, etc.

# Model Architecture

Uses multiple convolutional layers with batch normalization and max pooling.

Includes dropout layers to prevent overfitting.

Uses a dense layer with softmax activation for final classification.

Optimized using the Adam optimizer.

# Training Process

Data Preprocessing: Images are resized and augmented using ImageDataGenerator.

Model Training: The CNN model is trained with early stopping and checkpointing.

Evaluation: Model performance is assessed using accuracy and loss on the test set.

# Dependencies

Install the required dependencies using:

pip install tensorflow opencv-python matplotlib kaggle

Running the Model

Clone the repository or download the code.

Ensure the FER2013 dataset is placed in the correct directory.

Run the Jupyter Notebook or Python script to train the model.

Evaluate the model's performance using the test set.

# Limitations

The current model achieves ~67% accuracy but may require more computational resources for further improvement.

Additional hyperparameter tuning and architectural changes may enhance performance.

# Future Enhancements

Fine-tuning using transfer learning.

Implementing attention mechanisms to improve feature extraction.


# Acknowledgments

The FER2013 dataset from Kaggle.

TensorFlow and Keras for deep learning implementation.

OpenCV for image preprocessing.

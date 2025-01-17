"""
MNIST Handwritten Digit Classification
A comprehensive implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits
using the MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Setting random seeds ensures reproducibility across runs
# This is crucial for debugging and consistent results
np.random.seed(256)
tf.random.set_seed(256)

"""Data Loading and Initial Exploration"""
# The MNIST dataset is a widely used benchmark in machine learning
# It consists of 28x28 grayscale images of handwritten digits (0-9)
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Understanding our data dimensions helps verify the loading process
# The shape shows: (number_of_images, image_height, image_width)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

"""Data Preprocessing Phase"""
# Normalize pixel values from [0-255] to [0-1] range
# This step is crucial for neural networks to work effectively
# Smaller input values help prevent numerical instability during training
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to include a channel dimension
# CNNs expect input shape: (batch_size, height, width, channels)
# Even for grayscale images, we need a channel dimension of 1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoded format
# This is necessary for categorical crossentropy loss and multi-class classification
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

"""Visualization Functions"""
def plot_sample_digits(X, y, num_samples=5):
    """
    Visualize sample digits from the dataset with their corresponding labels.
    This helps in understanding the data we're working with and verifying
    the correct loading and preprocessing of images.
    
    Args:
        X: Array of images
        y: Array of labels (one-hot encoded)
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {np.argmax(y[i])}")
        plt.axis('off')
    plt.show()

# Display sample digits to verify data loading and preprocessing
print("\nDisplaying sample digits from training set:")
plot_sample_digits(X_train, y_train)

"""Model Architecture Design"""
def create_model():
    """
    Create and return a CNN model architecture designed for MNIST digit recognition.
    It depicts:
    Multiple convolutional layers for feature extraction
    Batch normalization for training stability
    Dropout for regularization
    Dense layers for final classification
    
    Returns:
        A compiled Keras Sequential model
    """
    model = models.Sequential([
        # First Convolutional Block
        # 32 filters help detect basic features like edges and curves
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),  # Stabilizes training by normalizing layer inputs
        layers.MaxPooling2D((2, 2)),  # Reduces spatial dimensions and computation
        layers.Dropout(0.25),  # Prevents overfitting by randomly dropping 25% of connections
        
        # Second Convolutional Block
        # 64 filters capture more complex patterns and combinations of features
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Transition to Dense Layers
        layers.Flatten(),  # Convert 2D feature maps to 1D vector
        layers.Dense(128, activation='relu'),  # Learn high-level combinations of features
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Higher dropout rate in dense layers
        layers.Dense(10, activation='softmax')  # Output probabilities for 10 digits
    ])
    return model

# Create and compile the model with appropriate optimizer and loss function
model = create_model()
model.compile(optimizer='adam',  # Adam optimizer adapts learning rate for each parameter
              loss='categorical_crossentropy',  # Standard loss for multi-class problems
              metrics=['accuracy'])  # Track accuracy during training

print("\nModel Summary:")
model.summary()

"""Model Training Phase"""
print("\nTraining the model...")
# The fit method handles the entire training process
# We use a validation split to monitor for overfitting
history = model.fit(X_train, y_train,
                    batch_size=128,  # Process 128 images at a time
                    epochs=10,  # Complete passes through the training data
                    validation_split=0.2,  # Use 20% of training data for validation
                    verbose=1)  # Show progress bar

"""Training Visualization"""
def plot_training_history(history):
    """
    Visualize the model's learning progress over time.
    This helps identify potential overfitting and verify proper training.
    
    Args:
        history: Training history object from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy progression shows how well the model learns
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss progression helps identify overfitting
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.show()

print("\nPlotting training history:")
plot_training_history(history)

"""Model Evaluation and Analysis"""
# Generate predictions on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Visualize confusion matrix to understand model's strengths and weaknesses
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

"""Error Analysis"""
def plot_misclassified_examples(X_test, y_test_classes, y_pred_classes, num_examples=5):
    """
    Visualize examples where the model made mistakes.
    This helps understand the types of errors the model makes and
    identify potential areas for improvement.
    
    Args:
        X_test: Test images
        y_test_classes: True labels
        y_pred_classes: Predicted labels
        num_examples: Number of misclassified examples to show
    """
    misclassified_idx = np.where(y_test_classes != y_pred_classes)[0]
    plt.figure(figsize=(15, 3))
    
    for i, idx in enumerate(misclassified_idx[:num_examples]):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_test_classes[idx]}\nPred: {y_pred_classes[idx]}')
        plt.axis('off')
    plt.show()

print("\nShowing examples of misclassified digits:")
plot_misclassified_examples(X_test, y_test_classes, y_pred_classes)

"""Model Persistence"""
# Save the trained model for future use
model.save('mnist_model.h5')
print("\nModel saved as 'mnist_model.h5'")

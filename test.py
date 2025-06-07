import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Addition of a 2 channel dimension to create RGB images
train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

# Undersample the training and test set to 500 balanced samples
num_samples = 90
train_indices = np.random.choice(np.where(train_labels == 0)[0], num_samples // 10, replace=False)
for i in range(1, 10):
    train_indices = np.concatenate((train_indices, 
                                    np.random.choice(np.where(train_labels == i)[0], num_samples // 10, replace=False)))
test_indices = np.random.choice(np.where(test_labels == 0)[0], num_samples // 10, replace=False)
for i in range(1, 10):
    test_indices = np.concatenate((test_indices, 
                                   np.random.choice(np.where(test_labels == i)[0], num_samples // 10, replace=False)))
train_images = train_images[train_indices]
train_labels = train_labels[train_indices]
test_images = test_images[test_indices]
test_labels = test_labels[test_indices]

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

# Define the CNN model
model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional layer
    layers.Conv2D(16, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(train_images, train_labels, epochs=100, batch_size=1,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
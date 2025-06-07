# Neural Network Training Module

The core of this project is a custom-built neural network training module implemented in C++. This module is responsible for constructing, training, and evaluating the neural network.

## Key Features

- **Layered Architecture:** The network is composed of convolutional, pooling, and fully connected layers, all implemented from scratch.
- **Forward and Backward Propagation:** The module handles both the forward pass (computing predictions) and backward pass (computing gradients and updating weights) without external libraries.
- **Loss Calculation:** Uses cross-entropy loss for classification tasks.
- **Optimization:** Implements SGD with Momentum, Adagrad, Adadelta, RMSProp, Adam for weight updates.
- **Model Saving/Loading:** After training, the model weights are saved to disk (`last.nn` and `best.nn`) for later inference or further training.
- **Evaluation:** The module evaluates accuracy on the validation/test set after each epoch, tracking the best-performing model.

## Achievements

- **MNIST with Dense Neural Network (DNN):** Achieved **90.23%** accuracy on the test set.
- **MNIST with Convolutional Neural Network (CNN):** Achieved **95.17%** accuracy on the test set.
- **CIFAR-10 with Convolutional Neural Network (CNN):** Achieved **81.2%** accuracy on the test set.

## Training Workflow

1. **Data Loading:** Reads and preprocesses the dataset.
2. **Model Initialization:** Sets up the architecture with user-defined parameters.
3. **Epoch Loop:** For each epoch:
   - Shuffles and batches the training data.
   - Performs forward and backward propagation for each batch.
   - Updates weights using the computed gradients.
   - Evaluates performance on the validation set.
   - Saves the best model based on validation accuracy.
4. **Result:** Outputs the final accuracy and saves the trained model weights.

## Customization

You can modify the network architecture, learning rate, batch size, and other hyperparameters directly in the source code to experiment with different configurations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

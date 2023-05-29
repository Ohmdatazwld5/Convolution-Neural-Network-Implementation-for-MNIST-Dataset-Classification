# Convolution-Neural-Network-Implementation-for-MNIST-Dataset-Classification
- Data Loading and Preprocessing:

->The MNIST dataset is loaded using tf.keras.datasets.mnist.load_data(), and the data is split into training and testing sets.

->The pixel values of the images in the training set are normalized by dividing them by 255.0.

->The training labels are one-hot encoded using tf.keras.utils.to_categorical()

- Model Architecture:

->A sequential model is created using Sequential() from Keras.

->The model starts with an input layer of shape (28, 28, 1) for the grayscale images.

->The model then adds a series of convolutional layers (Conv2D) with different filter sizes and activation functions (ReLU).

->Between the convolutional layers, a max pooling layer (MaxPool2D) is added to downsample the feature maps.

->A dropout layer (Dropout) is included to prevent overfitting.

->After the convolutional layers, the feature maps are flattened (Flatten) to be fed into the dense layers.

->Two dense layers (Dense) with ReLU activation are added, followed by a final dense layer with a softmax activation function for multiclass classification.

->The model summary is printed using model.summary().

- Model Compilation and Training:

->The model is compiled using stochastic gradient descent (SGD) optimizer, categorical cross-entropy loss function, and accuracy as the metric.

->The model is trained on the training data using model.fit(). The training data is batched (batch_size=64), and a validation split of 0.1 is used for validation during training. The training is performed for 12 epochs.

- Model Evaluation:

->The model is used to make predictions on the test data using model.predict().

->The predicted labels are obtained by taking the argmax along the predicted probabilities axis.

->The accuracy of the model is calculated using accuracy_score() from sklearn.metrics, comparing the predicted labels with the true labels.

->The confusion matrix is computed using confusion_matrix() from sklearn.metrics, comparing the predicted labels with the true labels.

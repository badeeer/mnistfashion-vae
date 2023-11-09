
# Introduction to VAE and Fashion MNIST

*Inspired by the book "Deep Learning with Python" by François Chollet*

The **Variational Autoencoder (VAE)** is a type of generative model used in machine learning and deep learning. It is particularly useful for tasks like data generation, image reconstruction, and feature learning. The **Fashion MNIST** dataset serves as a suitable candidate for applying VAE.

## Fashion MNIST Dataset

Fashion MNIST is a dataset comprising grayscale images of clothing and fashion items. It contains 60,000 training images and 10,000 test images across ten different classes, each representing a different fashion item such as shoes, dresses, or t-shirts. The dataset is widely used in machine learning for image classification tasks and, in this context, for VAE-based generative modeling.

## The Problem

The problem addressed here is to leverage a Variational Autoencoder to learn a compact, continuous latent space representation of the Fashion MNIST images. VAEs are designed to capture the underlying structure in data and enable the generation of new, similar data samples from this learned latent space.

## Import necessary libraries and modules

 - `import tensorflow as tf`: This line imports the TensorFlow library, which is a widely-used deep learning framework. It's necessary for building and training the machine learning model.

 - `from tensorflow.keras.datasets import fashion_mnist`: This line imports the Fashion MNIST dataset from the TensorFlow.keras.datasets module. Fashion MNIST contains grayscale images of various clothing items and accessories, making it a suitable dataset for machine learning.

- `from tensorflow.keras.models import Model`: This line imports the Model class from the Keras library. Keras is a high-level neural network API that operates on top of TensorFlow. The Model class will be used to define the architecture of the machine learning model.

 - `from tensorflow.keras import layers`: This line imports various layers and components from Keras. These layers are essential for constructing the neural network model, including dense layers and convolutional layers.

 - `from tensorflow import keras`: This line imports the Keras subpackage from TensorFlow. Keras is a user-friendly and high-level framework for creating and training neural networks. It integrates seamlessly with TensorFlow, a lower-level deep learning framework.

 - `import matplotlib.pyplot as plt`: This line imports the Matplotlib library, which is commonly used for data visualization. Matplotlib will be used to display images and plots during the training and evaluation of the machine learning model.

 - `import numpy as np`: This line imports the NumPy library, which is essential for numerical operations in Python. It supports arrays, matrices, and various mathematical operations, making it crucial for data processing in machine learning.

## Load the Fashion MNIST dataset


- `(x_train, _), (x_test, _) = fashion_mnist.load_data()`: This code segment loads the Fashion MNIST dataset using the `fashion_mnist.load_data()` function. The dataset is divided into training (`x_train`) and testing (`x_test`) sets. The corresponding labels are loaded but not used in this code snippet.


- `x_train = x_train.astype('float32') / 255`: These lines normalize the pixel values in the training dataset (`x_train`). First, the pixel values are cast to floating-point numbers to ensure accurate calculations. Then, each pixel's value is divided by 255, scaling them to the range [0, 1]. This normalization is a common preprocessing step in machine learning.

- `x_test = x_test.astype('float32') / 255`: Similar to the previous explanation, these lines normalize the pixel values in the testing dataset (`x_test`) by casting to float and scaling to the range [0, 1].

- `x_train = x_train[..., tf.newaxis]`: These lines add a channel dimension to the images in the training dataset. The `tf.newaxis` operation introduces a new dimension to represent grayscale images. Typically, grayscale images have a single channel (as opposed to RGB images, which have three channels). This step prepares the data for input to a model that expects images in this format.

- `x_test = x_test[..., tf.newaxis]`: Similar to the previous explanation, these lines add a channel dimension to the images in the testing dataset, ensuring consistency with the training dataset.

- `latent_dim = 200`: This line defines the dimension of the latent space, which is set to 200. In the context of a Variational Autoencoder (VAE), the latent space represents a compressed and continuous representation of the input data. The dimension of the latent space is a hyperparameter that can be adjusted based on the specific requirements of the model and the dataset.

## Create a CNN architecture

- `encoder_inputs = keras.Input(shape=(28, 28, 1))`: This line defines the input layer for the VAE encoder network. The `shape=(28, 28, 1)` indicates that the input images are expected to have a size of 28x28 pixels and a single channel (grayscale images).

- `x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)`: This code segment applies a convolutional layer with 32 filters, a kernel size of 3x3, ReLU activation function, and a stride of 2. It effectively reduces the spatial dimensions by half (strides=2) while keeping the output size the same due to the 'same' padding. This layer extracts features from the input images.

- `x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)`: Similar to the previous line, another convolutional layer is applied to further extract features. This layer has 64 filters and maintains the same spatial dimensions reduction.

- `x = layers.Flatten()(x)`: This line flattens the output from the convolutional layers into a one-dimensional vector. It prepares the feature representations for further processing.

- `x = layers.Dense(16, activation='relu')(x)`: A dense (fully connected) layer with 16 units and ReLU activation is applied. This layer performs additional feature mapping and dimension reduction.

- `z_mean = layers.Dense(latent_dim, name='z_mean')(x)`: This line connects a dense layer with `latent_dim` units, representing the mean of the latent space. The `name='z_mean'` parameter assigns a name to this layer.

- `z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)`: Similar to the previous line, another dense layer with `latent_dim` units represents the log-variance of the latent space. The `name='z_log_var'` parameter assigns a name to this layer.

- `encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')`: This line creates the encoder model by specifying the input layer (`encoder_inputs`) and the outputs, which are the mean and log-variance of the latent space (`[z_mean, z_log_var]`). The model is given the name 'encoder'.

## Sampler 

- `class Sampler(layers.Layer)`: This line defines a custom layer named `Sampler` that inherits from the `layers.Layer` class. This custom layer will be responsible for sampling points in the latent space.

- Inside the `Sampler` layer's `call` method, it first takes `z_mean` and `z_log_var` as inputs. `z_mean` represents the mean of the latent space, and `z_log_var` represents the log-variance.

- `epsilon = tf.random.normal(shape=tf.shape(z_mean))` generates random samples from a normal distribution with the same shape as `z_mean`. This is a key step in the reparameterization trick, allowing the model to sample latent points in a differentiable manner.

- The final line, `return z_mean + tf.exp(0.5 * z_log_var) * epsilon`, calculates the sampled latent variable `z`. It combines the mean `z_mean`, the exponential of half the log-variance, and the random noise `epsilon` to generate a sample from the latent space.

In the context of a VAE, this `Sampler` layer plays a crucial role in sampling latent points that can be used to generate new data or perform other tasks like reconstruction. It ensures that the sampling operation is differentiable, making it possible to train the model using gradient-based optimization methods.

## VAE decoder network

- `latent_inputs = keras.Input(shape=(latent_dim,))`: This line defines the input layer for the VAE decoder network. The input shape is specified as `(latent_dim,)`, where `latent_dim` represents the dimension of the latent space. This input will receive the sampled latent variables generated by the VAE encoder.

- `x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)`: A dense layer is applied to the input `latent_inputs` to map the latent space to a higher-dimensional space. The number `7 * 7 * 64` is used to adapt the dimension to the desired dimensions. The ReLU activation function is applied.

- `x = layers.Reshape((7, 7, 64))(x)`: This line reshapes the output from the previous layer to match the desired spatial dimensions of `(7, 7, 64)`. This is a common operation to prepare the data for transposed convolution layers.

- `x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)`: A transposed convolutional layer is applied with 64 filters, a kernel size of 3x3, ReLU activation, a stride of 2, and 'same' padding. This layer helps in upsampling the data and increasing spatial dimensions.

- `x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)`: Similar to the previous line, another transposed convolutional layer is applied with 32 filters, 3x3 kernel, ReLU activation, stride of 2, and 'same' padding.

- `decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)`: The final transposed convolutional layer generates the decoder outputs. It has 1 filter, a 3x3 kernel, a sigmoid activation function, and 'same' padding. In the context of Fashion MNIST, a single channel (1) is used to represent the reconstructed grayscale images.

- `decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')`: This line defines the decoder model. It takes the `latent_inputs` as input and produces `decoder_outputs`. The model is named 'decoder'.

This code defines the architecture of the VAE decoder network, which takes the sampled latent variables and maps them back to the space of reconstructed images. It's an essential part of the Variational Autoencoder (VAE) for tasks like image generation and reconstruction.

## VAE model

- `class VAE(keras.Model)`: This code defines a custom class named `VAE` that inherits from `keras.Model`. This class represents the Variational Autoencoder (VAE) model.

- `def __init__(self, encoder, decoder, **kwargs)`: The constructor for the `VAE` class takes two arguments, `encoder` and `decoder`, which are instances of the encoder and decoder models. It also accepts optional keyword arguments.

- `self.encoder = encoder`: This line assigns the provided `encoder` to the `self.encoder` attribute, allowing the VAE model to use the encoder to generate latent representations.

- `self.decoder = decoder`: Similarly, this line assigns the provided `decoder` to the `self.decoder` attribute, allowing the VAE model to use the decoder for image reconstruction.

- `self.sampler = Sampler()`: The VAE uses a `Sampler` layer to generate latent samples. This line initializes the `self.sampler` attribute.

- `self.total_loss_tracker`, `self.reconstruction_loss_tracker`, and `self.kl_loss_tracker`: These lines initialize three metrics to track the total loss, reconstruction loss, and KL divergence loss during training.

- `@property def metrics(self)`: This property defines the metrics to be tracked during training, which include total loss, reconstruction loss, and KL divergence loss.

- `def train_step(self, data)`: This method defines a single training step for the VAE. It takes a batch of input data as `data`.

- Inside the training step, it uses a `tf.GradientTape()` context to record operations for automatic differentiation. The encoder generates `z_mean` and `z_log_var` for the given data, and the `Sampler` layer is used to sample latent variables `z`. The decoder then reconstructs the data from `z`.

- Reconstruction loss is calculated using mean squared error between the input data and the reconstructed data.

- The KL divergence loss is also calculated based on the provided `z_mean` and `z_log_var`.

- The total loss is the sum of the reconstruction loss and KL divergence loss.

- Gradients are computed with respect to the total loss and used to update the model's trainable weights.

- The metrics for total loss, reconstruction loss, and KL divergence loss are updated.

- The function returns a dictionary containing the tracked losses.

- Finally, a VAE model is created by instantiating the `VAE` class with the provided `encoder` and `decoder`. It's compiled using the Adam optimizer, and it's trained on the `x_train` data for 100 epochs with a batch size of 128.

The provided code generates a grid of grayscale images using a Variational Autoencoder (VAE) and then displays the generated images in a plot. Here's an explanation of the code:

```python
# Define the parameters for image generation
n = 30
digit_size = 28  # Adjust the digit size to match the size of your grayscale images
figure = np.zeros((digit_size * n, digit_size * n), dtype=np.uint8)

grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)[::-1]
```

- `n` is set to 30, indicating that you want to generate a grid of 30x30 images.
- `digit_size` is set to 28, which defines the size of each digit in the grid. You should adjust this value to match the size of your grayscale images.
- `figure` is a NumPy array initialized to zeros and used to store the generated grid of images.
- `grid_x` and `grid_y` are arrays that define a grid of values in the latent space. In this example, the code creates a grid in a 2D latent space by varying `grid_x` and `grid_y` from -2 to 2. These values are used to sample from the latent space.


- Two nested loops iterate through the `grid_x` and `grid_y` values. In each iteration, a random sample `z_sample` is drawn from the latent space (in this case, 2 dimensions).
- The `z_sample` is then passed to the VAE's decoder, `vae.decoder`, to generate an image from the latent vector. This image is stored in `x_decoded`.
- The `x_decoded` image is rescaled to 8-bit grayscale (0-255) by multiplying by 255 and converting it to `np.uint8` format.
- The rescaled image is placed in the `figure` array at the corresponding position in the grid.

Finally, the code creates a plot to display the generated grayscale images:


## Define Parameters:

- `n` is set to 30, which represents the number of images in each row and column of the grid.
- `digit_size` is set to 28, which is the size of each grayscale image. You should adjust this to match the size of the grayscale images you are working with.
- `figure` is initialized as a NumPy array of zeros with dimensions `(digit_size * n, digit_size * n)`. This array will hold the generated images.

## Grid Creation:

- `grid_x` and `grid_y` are created using NumPy's `linspace` function. These arrays represent the coordinates in the latent space that will be used for sampling.

## Sampling and Image Generation:

- The code then enters a nested loop that iterates over the `grid_x` and `grid_y` coordinates.
- For each combination of `xi` and `yi`, it samples from the latent space using `np.random.normal`. The `latent_dim` is assumed to be defined elsewhere in your code and represents the dimensionality of the latent space (usually 2 for a 2D latent space).
- It then passes the sampled latent vector `z_sample` to the VAE's decoder model (`vae.decoder.predict(z_sample)`) to generate an image. The decoder maps the latent vector back to an image in this step.
- The generated image is then scaled and converted to an 8-bit grayscale image by multiplying by 255 and taking only the first channel (assuming it's a grayscale image).

## Populating the figure Array:

- The generated grayscale image is placed in the `figure` array at the appropriate position within the grid.

## Display the Generated Images:

- A plot is created using Matplotlib with a size of 10x10 inches.
- The `figure` array is displayed as an image using `plt.imshow`. The colormap is set to grayscale using `cmap='gray'`.
- `plt.axis("off")` is used to hide the axis.
- Finally, `plt.show()` is called to display the grid of generated grayscale images.


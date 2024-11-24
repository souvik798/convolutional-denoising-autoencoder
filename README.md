# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The goal is to design a convolutional autoencoder that can learn to remove noise from grayscale images. The model will take noisy images as input, compress them into a latent space, and then reconstruct a denoised version of the image. This problem is common in image processing tasks like noise reduction, compression, and anomaly detection.
![image](https://github.com/user-attachments/assets/d58355a0-ced9-4c08-9ff8-28f64fd8f946)


## Convolution Autoencoder Network Model

![image](https://github.com/user-attachments/assets/a3afc089-cac0-4c78-b3cf-6aad751d8029)

## DESIGN STEPS

STEP 1: Data Preprocessing
Load and normalize the MNIST dataset, add random Gaussian noise, and split it into training and testing sets.

STEP 2: Build the Convolutional Autoencoder Model
Create the encoder with convolutional layers and MaxPooling2D, and the decoder with convolutional layers and UpSampling2D, ending with a Conv2D layer to reconstruct denoised images.

STEP 3: Model Compilation and Training
Compile the model using the Adam optimizer and binary_crossentropy loss, then train it with noisy inputs and clean targets.

STEP 4: Model Evaluation and Visualization
Evaluate the model on noisy test images and visualize the denoised outputs alongside the original clean images.

## PROGRAM
### Name: SOUVIK KUNDU  
### Register Number: 212221230105

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# Input layer with shape (28, 28, 1) for grayscale images of size 28x28
input_img = keras.Input(shape=(28, 28, 1))

# Encoder
# First convolutional layer followed by ReLU activation
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
# Max pooling layer to downsample the feature maps
x = layers.MaxPooling2D((2, 2), padding='same')(x)

# Second convolutional layer followed by ReLU activation
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# Max pooling layer to further downsample the feature maps
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Encoder output dimension is (7, 7, 64)

# Decoder
# First upsampling layer to upsample the feature maps
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)

# Second upsampling layer to further upsample the feature maps
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)

# Output layer to reconstruct the input image, using sigmoid activation
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Define the autoencoder model
autoencoder = keras.Model(input_img, decoded)
print('Name: Bhargava          Register Number: 212221040029        ')
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
print('Name:           Register Number:        ')
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss and Validation Loss vs Epochs')

# Adding a legend
plt.legend()

# Show the plot
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![reew](https://github.com/user-attachments/assets/e6f28f0e-362e-41ce-ba03-238fe7a009d9)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/user-attachments/assets/a3760c99-0b94-4e7c-ba78-211788e80453)
![image](https://github.com/user-attachments/assets/b1f52783-dbdc-41de-b97c-156698b5d7c3)


## RESULT
The convolutional autoencoder was trained on noisy MNIST images for denoising. The training and validation loss steadily decreased over the epochs, indicating that the model successfully learned to reconstruct clean images from noisy inputs. The final plot of **Training Loss** vs **Validation Loss** shows convergence, with both losses closely aligning, suggesting that the model generalizes well without overfitting.

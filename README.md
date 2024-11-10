# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
An autoencoder is an unsupervised neural network that encodes input images into lower-dimensional representations and decodes them back, aiming for identical outputs. We use the MNIST dataset, consisting of 60,000 handwritten digits (28x28 pixels), to train a convolutional neural network for digit classification. The goal is to accurately classify each digit into one of 10 classes, from 0 to 9.

## Convolution Autoencoder Network Model
![image](https://github.com/user-attachments/assets/bb6e4510-0965-40cc-9f4d-8c1b0514e2f4)

## DESIGN STEPS

### STEP 1:
Load the MNIST dataset and preprocess it by normalizing the pixel values to a range of 0 to 1. Then, add Gaussian noise to the dataset to create noisy images.

### STEP 2:
Build the autoencoder architecture using a series of convolutional and pooling layers for the encoder, followed by convolutional and upsampling layers for the decoder.

### STEP 3:
Compile the model with an appropriate loss function (binary_crossentropy) and optimizer (adam). Train the model using the noisy images as input and the original images as targets. Use the validation set to monitor the performance during training.


## PROGRAM
### Name:SOUVIK KUNDU
### Register Number:212221230105
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, utils, models
from tensorflow.keras.datasets import mnist
(xtrain,_), (xtest,_) = mnist.load_data()
xtrain.shape
xtrainS = xtrain.astype('float32') / 255.
xtestS = xtest.astype('float32') / 255.
xtrainS = np.reshape(xtrainS,(len(xtrainS),28,28,1))
xtestS = np.reshape(xtestS,(len(xtestS),28,28,1))
nf = 0.5
xtrainN=xtrainS+nf*np.random.normal(loc=0.0,scale=1.0,size=xtrainS.shape)
xtestN=xtestS+nf*np.random.normal(loc=0.0,scale=1.0,size=xtestS.shape)
xtrainN = np.clip(xtrainN, 0., 1.)
xtestN = np.clip(xtestN, 0., 1.)
n = 10
plt.figure(figsize=(20,2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(xtestN[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
inp_img=keras.Input(shape=(28,28,1))
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(inp_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoder=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(encoder)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(3,3),activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
decoder=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
model=keras.Model(inp_img,decoder)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(xtrainN,xtrainS,epochs=2,batch_size=128,shuffle=True,validation_data=(xtestN,xtestS))
metrics=pd.DataFrame(model.history.history)
plt.figure(figsize=(7,2.5))
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.title('Training Loss vs. Validation Loss\nJEEVAGOWTHAM S - 212222230053')
decodeimg=model.predict(xtestN)
def display_images(xtestS, xtestN, decodeimg, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        for j, img in enumerate([xtestS,xtestN,decodeimg]):
            ax = plt.subplot(3, n, i + 1 + j * n)
            plt.imshow(img[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
    plt.show()
display_images(xtestS, xtestN, decodeimg)

```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-11-07 144325](https://github.com/user-attachments/assets/5433d655-e6c3-4738-82a9-6c52b1f1e671)


### Original vs Noisy Vs Reconstructed Image
![Screenshot 2024-11-07 144239](https://github.com/user-attachments/assets/d2b62180-f3ec-4085-a3bc-f2836443d66b)



## RESULT:
Thus,The convolutional autoencoder for image denoising application is successfully executed.


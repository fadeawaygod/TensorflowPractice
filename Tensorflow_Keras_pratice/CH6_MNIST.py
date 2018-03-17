import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.utils import np_utils

np.random.seed(10)

(train_data, train_label), (test_data, test_label) = mnist.load_data()

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

#plot_image(train_data[0])

#reshape 2d to 1d
train_data = train_data.reshape(60000, 784).astype('float32')
test_data = test_data.reshape(10000, 784).astype('float32')

#normalization
train_data /= 255
test_data /= 255

#one-hot-encoding
train_label_1hot = np_utils.to_categorical(train_label)
test_label_1hot = np_utils.to_categorical(test_label)
pass

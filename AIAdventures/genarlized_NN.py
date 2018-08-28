import tflowtools as TFT
import numpy as np
import matplotlib.pyplot as plt # In case of plotting
import tensorflow as tf
from functools import reduce
import os
import mnist_basics as mnst
import pylab

#First, we try to create a simple Neural network for the MNIST dataset

def read_file(file):
    """
    param file: path to file
    """
    return open(file, "r")



def scale_to_mean(data):
    data_mean = reduce(lambda x, y: x + y, data) / len(data)
    #Or use np.mean
    data_mean = np.mean(data) # Mean of all data points in list.
    data_std = np.std(data) # Standard deviation of all data points in list.
    return [(element-data_mean) / data_std for element in data]

def scale_to_extremes(data):
    min_data_point, max_data_point = np.min(data), np.max(data)
    domain = max_data_point - min_data_point
    return [(element - min_data_point) / domain for element in data]

def plot_single_image(img):
    plt.figure()
    plt.gray()
    pylab.imshow(img)

def plot_images(image_array, width = 5, height = 5):
    plt.figure()
    for i, image in enumerate(image_array):
        print(image.shape)
        pylab.subplot(height, width, i + 1); pylab.axis('off'); pylab.imshow(image);



if __name__== "__main__":
    #load mnist dataset
    images, labels = mnst.load_mnist()
    print(images[0])

    plot_single_image(images[0])
    plot_images(images[:50], width =10, height = 5)
    plt.show()
    test_list = [5, 6, 7, 155, 3, 199, 243]
    print(scale_to_mean(test_list))
    print(scale_to_extremes(test_list))
    #load_dataset(test_set)

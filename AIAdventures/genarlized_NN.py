import tflowtools as TFT
import numpy as np
import matplotlib.pyplot as plt # In case of plotting
import tensorflow as tf
from functools import reduce
import os
import mnist_basics as mnst
import pylab

activation_functions = {"relu" : tf.nn.relu, "sigmoid" : tf.nn.sigmoid, "tanh" : tf.nn.tanh}


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





#Generate network dimensions
def initialize_network_dimensions(layers):
    for layer in layers:
        #Add neurons
        break
    return

def get_activation_function(activation):
    try:
        return activation_functions[activation]
    except Exception as e:
        raise ValueError("Choose a valid activation function")

class general_NN():
    def __init__(dimensions,
                 hidden_activation_function,
                 oaf,
                 loss_func,
                 lr,
                 weight_range,
                 optimizer,
                 data_source,
                 case_fraction,
                 validation_fraction,
                 vint,
                 test_fraction,
                 minibatch_size,
                 map_batch_size,
                 steps,
                 map_layers,
                 map_dendrograms,
                 display_weights,
                 display_biases):
        import inspect
        args, _, _, _ = inspect.getargvalues(inspect.currentframe())
        print("args", args)
        for arg in args:
            setattr(self, arg, locals[arg])


    #master function



if __name__== "__main__":
    args = tuple([0]*19)
    my_nn = general_NN(args)
    """
    get_activation_function("hei")
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
    """

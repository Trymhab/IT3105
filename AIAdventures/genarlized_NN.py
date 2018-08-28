import tflowtools as TFT
import numpy as np
import matplotlib.pyplot as plt # In case of plotting
import tensorflow as tf
from functools import reduce


#First, we try to create a simple Neural network for the MNIST dataset

def load_dataset(file):
    return

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

if __name__== "__main__":
    test_list = [5, 6, 7, 155, 3, 199, 243]
    print(scale_to_mean(test_list))
    print(scale_to_extremes(test_list))

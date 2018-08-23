import tensorflow as tf
import numpy as np
import math as m
import matplotlib.pyplot as plt

def get_inputs():
    print("Input format: Network dimensions, Hidden activation function, Output activation, ...)
    tuple_string = input("Input: ")
    return tuple_string.split(", ")

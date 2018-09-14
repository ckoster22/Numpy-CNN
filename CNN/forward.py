'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
import numpy as np


def convolution(image, filt, bias, stride=1):
    '''
    Confolves `filt` over `image` using stride `stride`
    '''
    (num_filters, num_channels_filter, kernel_size, _) = filt.shape
    num_channels, in_dim, _ = image.shape

    # calculate output dimensions
    out_dim = int((in_dim - kernel_size) / stride) + 1

    assert num_channels == num_channels_filter, "Number of filter channels must match number of input image channels"

    out = np.zeros((num_filters, out_dim, out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    for filter_index in range(num_filters):
        curr_y = out_y = 0
        curr_filter = filt[filter_index]
        while curr_y + kernel_size <= in_dim:
            curr_x = out_x = 0
            while curr_x + kernel_size <= in_dim:
                out[filter_index, out_y, out_x] = np.sum(
                    curr_filter * image[:, curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size]) + bias[filter_index]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return out


def maxpool(image, kernel_size=2, stride=2):
    '''
    Downsample `image` using kernel size `kernel_size` and stride `stride`
    '''
    num_channels, image_height, image_width = image.shape

    out_height = int((image_height - kernel_size) / stride) + 1
    out_width = int((image_width - kernel_size) / stride) + 1

    downsampled = np.zeros((num_channels, out_height, out_width))
    for channel_index in range(num_channels):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + kernel_size <= image_height:
            curr_x = out_x = 0
            while curr_x + kernel_size <= image_width:
                downsampled[channel_index, out_y, out_x] = np.max(
                    image[channel_index, curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size])
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
    return downsampled


def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))

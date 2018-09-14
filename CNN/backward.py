'''
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''

import numpy as np

from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################
        
def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (num_filters, num_channels, kernel_size, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    filter_loss_grad = np.zeros(filt.shape)
    dbias = np.zeros((num_filters,1))
    for filter_index in range(num_filters):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + kernel_size <= orig_dim:
            curr_x = out_x = 0
            while curr_x + kernel_size <= orig_dim:
                # loss gradient of filter (used to update the filter)
                filter_loss_grad[filter_index] += dconv_prev[filter_index, out_y, out_x] * conv_in[:, curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size] += dconv_prev[filter_index, out_y, out_x] * filt[filter_index] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[filter_index] = np.sum(dconv_prev[filter_index])
    
    return dout, filter_loss_grad, dbias



def maxpoolBackward(dpool, orig, kernel_size, stride):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (num_channels, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for channel_index in range(num_channels):
        curr_y = out_y = 0
        while curr_y + kernel_size <= orig_dim:
            curr_x = out_x = 0
            while curr_x + kernel_size <= orig_dim:
                # obtain index of largest value in input for current window
                (y_index, x_index) = maxValIndices(orig[channel_index, curr_y:curr_y+kernel_size, curr_x:curr_x+kernel_size])
                dout[channel_index, curr_y+y_index, curr_x+x_index] = dpool[channel_index, out_y, out_x]
                
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        
    return dout

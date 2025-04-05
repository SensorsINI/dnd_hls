import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from scipy import signal
import math

from plotmlpfkernels import plot_kernels

def gabor_initializer_1d_patch(patch_height=7, patch_width=7, num_filters=20):
    """
    Generates a Keras initializer for a Dense layer that initializes the kernels
    as a set of rotated 1D Gabor filters. The input to the Dense layer is
    assumed to be a flattened 2D image patch, with the first half holding relative event 
    age and the 2nd half holding signed event polarity.

    Args:
        patch_height (int): Height of the input image patch. Must be odd integer
        patch_width (int): Width of the input image patch.
        num_filters (int): The number of Gabor filters (and thus the number
                           of output units in the Dense layer). Must be even for same # ON and OFF kernels.
                           There will be two actual kernels for each of num_filters (age and polarity kernels) 
                           that are concatenated to a single 1d kernel.

    Returns:
        callable: A Keras initializer function that can be used for the `kernel_initializer`
                  argument of a `Dense` layer.
    """
    
        #     orientations (list or np.ndarray): A list of orientations (in degrees)
        #                                    for the Gabor filters. The length
        #                                    should be equal to num_filters.
        # frequencies (list or np.ndarray): A list of spatial frequencies for the
        #                                     Gabor filters. The length should be
        #                                     equal to num_filters.
        # sigmas (list or np.ndarray): A list of standard deviations (sigma_x, sigma_y)
        #                              for the Gaussian envelope of the Gabor filters.
        #                              Each element can be a tuple/list of two floats
        #                              or a single float (in which case sigma_x = sigma_y).
        #                              The length should be equal to num_filters.
        # gammas (list or np.ndarray): A list of aspect ratios (gamma = sigma_y / sigma_x)
        #                              of the Gaussian envelope. The length should be
        #                              equal to num_filters.
        # phis (list or np.ndarray): A list of phase offsets (in radians) for the
        #                            sinusoidal component of the Gabor filters.
        #                            The length should be equal to num_filters.

    
    # , orientations, frequencies, sigmas, gammas, phis

    orientations = np.linspace(0,360,endpoint=False,num=num_filters)
    frequency = 0.3
    sigma = 1
    gamma =2
   

    def _gabor_initializer_1d_patch(shape, dtype=None):
        ''' Generates weights for hidden first layer of MLP given input shape.
        The input shape is e.g. (98,) which is 2*49=2*7*7.
        
        :param shape: input shape
        :returns: a set of kernels with shape (98,20) for 20 hidden units
        '''
        if shape != [patch_height * patch_width*2, num_filters]:
            raise ValueError(f"Expected kernel shape {(patch_height * patch_width*2, num_filters)}, but got {shape}")
        
        if patch_width%2==0:
            raise ValueError(f'patch_width={patch_width} must be odd so kernel is symmetric')

        if num_filters%2!=0:
            raise ValueError(f'num_filters={num_filters} must be even so kernels are half on and half off')

        kernels = np.zeros(shape, dtype=np.float32)

        for i in range(num_filters):
            theta_rad = np.deg2rad(orientations[i])

            sigma_x = sigma/gamma
            sigma_y = sigma*gamma

            npt=patch_width//2 # e.g. 7->3
            pts=list(range(-npt,npt+1))
            # Create 2D Gabor filter
            x, y = np.meshgrid(pts,pts)

            x_theta = x * np.cos(theta_rad) + y * np.sin(theta_rad)
            y_theta = -x * np.sin(theta_rad) + y * np.cos(theta_rad)

            gb = np.exp(-0.5 * ((x_theta / sigma_x)**2 + (y_theta / sigma_y)**2)) * np.cos(2 * np.pi * frequency * x_theta)
            
            if i>=num_filters/2:
                gb=-gb

            # double the kernel for age and polarity
            k=gb.flatten()
            k=np.concatenate([k,k],axis=None)
            
            # Flatten the 2D Gabor filter and assign it to the kernel
            kernels[:, i] = k

            # Normalize the filter (optional, but often helpful)
            kernels[:, i] /= np.linalg.norm(kernels[:, i]) + 1e-8

        print(f' built gabor initial kernels with shape {kernels.shape}')
        return tf.constant(kernels, dtype=dtype)

    return _gabor_initializer_1d_patch # note that calling gabor_initializer_1d_patch() returns the *function* _gabor_initializer_1d_patch

if __name__ == '__main__':
    # Example usage:
    patch_height = 7
    patch_width = 7
    num_filters = 20

    gabor_init = gabor_initializer_1d_patch(
        patch_height=patch_height,
        patch_width=patch_width,
        num_filters=num_filters
    )

    # Create a simple Keras model with a Dense layer using the Gabor initializer
    input_shape=[patch_height * patch_width *2,]
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(units=num_filters,
                           kernel_initializer=gabor_init,
                           use_bias=False)  # Often Gabor filters are used without bias
    ])

    plot_kernels(model,None,'initial_kernals')

    # # Get the initialized weights of the Dense layer
    # initialized_weights = model.layers[0].get_weights()[0]
    # print("Shape of initialized weights:", initialized_weights.shape)  # Should be (98,20)

    # # You can visualize the initialized filters (optional)
    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(2, num_filters, figsize=(12, 3))
    # for i in range(num_filters):
    #     row=0
    #     col=i
    #     linear_kernel=initialized_weights[:, i]
    #     gabor_filter_2d = linear_kernel.reshape(2, patch_height, patch_width)
    #     axes[row,col].imshow(gabor_filter_2d[0,:,:], cmap='gray')
    #     axes[row,col].set_title(f'age {i}')
    #     axes[row,col].axis('off')
    #     axes[row+1,col].imshow(gabor_filter_2d[1,:,:], cmap='gray')
    #     axes[row+1,col].set_title(f'pol {i}')
    #     axes[row+1,col].axis('off')
    # plt.tight_layout()
    # plt.show()
import numpy as np
from scipy.signal import convolve2d

def generate_circular_kernel(radius):
    """
    Generate a circular kernel with a given radius.
    
    Parameters:
    radius (int): The radius of the circular kernel.

    Returns:
    numpy.ndarray: A 2D array representing the circular kernel.
    """
    # Determine the size of the kernel
    size = 2 * radius + 1

    # Create a grid of coordinates
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

    # Create the circular mask
    mask = x**2 + y**2 <= radius**2

    # Create the kernel
    kernel = np.zeros((size, size))
    kernel[mask] = 1.0

    return kernel

def MNCA_Test(input_buffer):

    kernel1 = generate_circular_kernel(3)
    kernel2 = generate_circular_kernel(8)

    # Perform the convolutions
    result1 = convolve2d(input_buffer, kernel1 / np.sum(kernel1), mode='same', boundary='wrap')
    result2 = convolve2d(input_buffer, kernel2 / np.sum(kernel2), mode='same', boundary='wrap')

    # Apply conditional update functions
    condition3 = (result2 > 0.12) & (result2 < 0.42)
    update3 = np.where(condition3, input_buffer - 0.14, input_buffer)

    condition1 = (result1 > 0.22) & (result1 < 0.34)
    update1 = np.where(condition1, update3 + 0.12, update3 - 0.02)

    condition2 = (result1 > 0.32) & (result1 < 0.4)
    update2 = np.where(condition2, update1 + 0.08, update1 - 0.02)

    clamped_result = np.clip(update2, 0.0, 1.0)

    return clamped_result


def simple_convolve(input_buffer):
    """
    Perform a simple convolution on the input buffer and clamp values between 0.0 and 1.0.
    """

    # Define a simple kernel, e.g., a blur or edge detection kernel
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9.0  # Example: Averaging (blur) kernel

    # Perform the convolution
    result = convolve2d(input_buffer, kernel, mode='same', boundary='wrap')

    # Clamp the result between 0.0 and 1.0
    clamped_result = np.clip(result, 0.0, 1.0)

    return clamped_result
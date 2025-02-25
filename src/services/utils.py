import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

#TODO: remove ipywidgets and matplotlib from this file

#TODO: clean this file and prep it for use with sidebar  

# FFT Filter (Low-pass filter in frequency domain)
def apply_fft_filter(image, cutoff_frequency):
    """
    Apply FFT-based low-pass filter to the image.

    Parameters:
    - image: Grayscale image (numpy array)
    - cutoff_frequency: Radius of the circular mask (pixels)

    Returns:
    - Filtered image (uint8)
    """
    # Compute FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create circular mask (1 inside radius, 0 outside)
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols), np.float32)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 1

    # Apply mask and inverse FFT
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # Normalize to 0-255 and convert to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

# Gaussian Filter
def apply_gaussian_filter(image, kernel_size, sigma):
    """
    Apply Gaussian blur to the image.

    Parameters:
    - image: Grayscale image (numpy array)
    - kernel_size: Size of the Gaussian kernel (must be odd)
    - sigma: Standard deviation of the Gaussian

    Returns:
    - Filtered image (uint8)
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered_image

# Sobel Filter
def apply_sobel_filter(image, kernel_size):
    """
    Apply Sobel edge detection to the image.

    Parameters:
    - image: Grayscale image (numpy array)
    - kernel_size: Size of the Sobel kernel (must be odd)

    Returns:
    - Filtered image (uint8)
    """
    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to 0-255 and convert to uint8
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

# Interactive Functions for Jupyter with Sliders
def interactive_gaussian(image):
    """
    Interactive Gaussian filter with sliders for kernel_size and sigma.

    Parameters:
    - image: Grayscale image (numpy array)
    """
    def apply_and_display(kernel_size, sigma):
        filtered = apply_gaussian_filter(image, kernel_size, sigma)
        plt.imshow(filtered, cmap='gray')
        plt.axis('off')
        plt.show()

    kernel_size_slider = widgets.IntSlider(min=1, max=31, step=2, value=5, description='Kernel Size')
    sigma_slider = widgets.FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0, description='Sigma')

    interact(apply_and_display, kernel_size=kernel_size_slider, sigma=sigma_slider)

def interactive_sobel(image):
    """
    Interactive Sobel filter with slider for kernel_size.

    Parameters:
    - image: Grayscale image (numpy array)
    """
    def apply_and_display(kernel_size):
        filtered = apply_sobel_filter(image, kernel_size)
        plt.imshow(filtered, cmap='gray')
        plt.axis('off')
        plt.show()

    kernel_size_slider = widgets.IntSlider(min=1, max=31, step=2, value=3, description='Kernel Size')

    interact(apply_and_display, kernel_size=kernel_size_slider)

def interactive_fft(image):
    """
    Interactive FFT filter with slider for cutoff_frequency.

    Parameters:
    - image: Grayscale image (numpy array)
    """
    def apply_and_display(cutoff_frequency):
        filtered = apply_fft_filter(image, cutoff_frequency)
        plt.imshow(filtered, cmap='gray')
        plt.axis('off')
        plt.show()

    max_radius = min(image.shape) // 2
    cutoff_slider = widgets.IntSlider(min=1, max=max_radius, step=1, value=30, description='Cutoff Frequency')

    interact(apply_and_display, cutoff_frequency=cutoff_slider)

# Function to Stack Filters (Optional)
def apply_filters(image, filter_list):
    """
    Apply a sequence of filters to the image.

    Parameters:
    - image: Grayscale image (numpy array)
    - filter_list: List of tuples (filter_name, params)

    Returns:
    - Filtered image (uint8)
    """
    result = image.copy()
    for filter_name, params in filter_list:
        if filter_name == 'gaussian':
            result = apply_gaussian_filter(result, **params)
        elif filter_name == 'sobel':
            result = apply_sobel_filter(result, **params)
        elif filter_name == 'fft':
            result = apply_fft_filter(result, **params)
    return result
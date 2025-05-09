import cv2
import numpy as np

# Remove matplotlib and ipywidgets imports as they're not needed
# import matplotlib.pyplot as plt
# from ipywidgets import interact, widgets

# Basic image conversion functions


def convert_to_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = convert_to_grayscale(image)

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
    - image: Image (numpy array)
    - kernel_size: Size of the Gaussian kernel (must be odd)
    - sigma: Standard deviation of the Gaussian

    Returns:
    - Filtered image (uint8)
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply filter
    if len(image.shape) > 2:
        filtered_image = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), sigma)
    else:
        filtered_image = cv2.GaussianBlur(
            image, (kernel_size, kernel_size), sigma)

    return filtered_image

# Sobel Filter


def apply_sobel_filter(image, kernel_size):
    """
    Apply Sobel edge detection to the image.

    Parameters:
    - image: Image (numpy array)
    - kernel_size: Size of the Sobel kernel (must be odd)

    Returns:
    - Filtered image (uint8)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to 0-255 and convert to uint8
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 3 channels if input was color image
    if len(image.shape) > 2:
        result = cv2.cvtColor(magnitude.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return result

    return magnitude.astype(np.uint8)


# Median Filter
def apply_median_filter(image, kernel_size):
    """
    Apply median filter to the image.

    Parameters:
    - image: Image (numpy array)
    - kernel_size: Size of the kernel (must be odd)

    Returns:
    - Filtered image (uint8)
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(image, kernel_size)

# Bilateral Filter


def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    """
    Apply bilateral filter to the image.

    Parameters:
    - image: Image (numpy array)
    - d: Diameter of each pixel neighborhood
    - sigma_color: Filter sigma in the color space
    - sigma_space: Filter sigma in the coordinate space

    Returns:
    - Filtered image (uint8)
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Laplacian Filter


def apply_laplacian_filter(image, kernel_size):
    """
    Apply Laplacian filter to the image.

    Parameters:
    - image: Image (numpy array)
    - kernel_size: Size of the Laplacian kernel (must be odd)

    Returns:
    - Filtered image (uint8)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    # Apply Laplacian filter
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)

    # Normalize to 0-255 and convert to uint8
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 3 channels if input was color image
    if len(image.shape) > 2:
        result = cv2.cvtColor(lap.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return result

    return lap.astype(np.uint8)

# Canny Edge Detector


def apply_canny_filter(image, threshold1, threshold2):
    """
    Apply Canny edge detection to the image.

    Parameters:
    - image: Image (numpy array)
    - threshold1: First threshold for the hysteresis procedure
    - threshold2: Second threshold for the hysteresis procedure

    Returns:
    - Edge image (uint8)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Convert to 3 channels if input was color image
    if len(image.shape) > 2:
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result

    return edges
# Morphological Operations


def apply_morphological_operation(image, operation, kernel_size):
    """
    Apply morphological operation to the image.

    Parameters:
    - image: Image (numpy array)
    - operation: Type of morphological operation ('dilate', 'erode', 'open', 'close')
    - kernel_size: Size of the structuring element

    Returns:
    - Processed image (uint8)
    """
    # Convert to grayscale if needed for certain operations
    if operation in ['erode', 'dilate', 'open', 'close'] and len(image.shape) > 2:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy() if len(image.shape) <= 2 else image.copy()

    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply operation
    if operation == 'dilate':
        result = cv2.dilate(gray, kernel, iterations=1)
    elif operation == 'erode':
        result = cv2.erode(gray, kernel, iterations=1)
    elif operation == 'open':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    else:
        result = gray  # Default to original image if operation not recognized

    # Convert back to 3 channels if input was color image
    if len(image.shape) > 2 and len(result.shape) <= 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result


# Kuwahara Filter (Artistic smoothing that preserves edges)
def apply_kuwahara_filter(image, kernel_size):
    """
    Apply Kuwahara filter for artistic smoothing while preserving edges.

    Parameters:
    - image: Image (numpy array)
    - kernel_size: Size of the kernel (must be odd and ≥5)

    Returns:
    - Filtered image (uint8)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    # kernel_size should be odd and >=5
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 5:
        kernel_size = 5

    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            sub_size = pad + 1
            regions = [
                window[0:sub_size, 0:sub_size],
                window[0:sub_size, -sub_size:],
                window[-sub_size:, 0:sub_size],
                window[-sub_size:, -sub_size:]
            ]
            region_means = [np.mean(r) for r in regions]
            region_vars = [np.var(r) for r in regions]
            output[i, j] = region_means[np.argmin(region_vars)]

    # Convert back to 3 channels if input was color image
    if len(image.shape) > 2:
        output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return output.astype(np.uint8)

# Cartoon Filter


def apply_cartoon_filter(image, bilateral_d, sigma_color, sigma_space, edge_threshold1, edge_threshold2):
    """
    Apply cartoon effect combining edge detection and bilateral filtering.

    Parameters:
    - image: Image (numpy array)
    - bilateral_d: Diameter of each pixel neighborhood for bilateral filter
    - sigma_color: Filter sigma in the color space
    - sigma_space: Filter sigma in the coordinate space
    - edge_threshold1: First threshold for Canny edge detection
    - edge_threshold2: Second threshold for Canny edge detection

    Returns:
    - Cartoon-style image (uint8)
    """
    # For color images
    if len(image.shape) > 2:
        # Apply bilateral filter for smoothing
        color = cv2.bilateralFilter(
            image, bilateral_d, sigma_color, sigma_space)

        # Convert to grayscale for edge detection
        gray = convert_to_grayscale(image)

        # Detect edges with Canny
        edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)

        # Invert edges and convert to 3 channels
        edges_inv = cv2.bitwise_not(edges)
        edges_inv_3ch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

        # Combine edges with the smoothed image
        return cv2.bitwise_and(color, edges_inv_3ch)
    else:
        # Grayscale image - simple version
        color = cv2.bilateralFilter(
            image, bilateral_d, sigma_color, sigma_space)
        edges = cv2.Canny(image, edge_threshold1, edge_threshold2)
        edges_inv = cv2.bitwise_not(edges)
        return cv2.bitwise_and(color, color, mask=edges_inv)

# Pencil Sketch Filter


def apply_pencil_sketch(image):
    """
    Apply pencil sketch effect to the image.

    Parameters:
    - image: Image (numpy array)

    Returns:
    - Sketch-style image (uint8)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = convert_to_grayscale(image)
    else:
        gray = image.copy()

    # Invert the grayscale image
    inverted = 255 - gray

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred = 255 - blurred

    # Divide the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)

    # Convert back to 3 channels if input was color image
    if len(image.shape) > 2:
        sketch = cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return sketch.astype(np.uint8)

# Function to Stack Filters (Optional)


# Update the apply_filters function to include new filters
def apply_filters(image, filter_list):
    """
    Apply a sequence of filters to the image.

    Parameters:
    - image: Image (numpy array)
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
        elif filter_name == 'median':
            result = apply_median_filter(result, **params)
        elif filter_name == 'bilateral':
            result = apply_bilateral_filter(result, **params)
        elif filter_name == 'laplacian':
            result = apply_laplacian_filter(result, **params)
        elif filter_name == 'canny':
            result = apply_canny_filter(result, **params)
        elif filter_name == 'morphological':
            result = apply_morphological_operation(result, **params)
        elif filter_name == 'kuwahara':
            result = apply_kuwahara_filter(result, **params)
        elif filter_name == 'cartoon':
            result = apply_cartoon_filter(result, **params)
        elif filter_name == 'pencil':
            result = apply_pencil_sketch(result)
    return result

# --- Feature Detection and Homography ---


def detect_and_describe_features(image, method='sift', color_mode='grayscale'):
    """
    Detect and describe features using SIFT, ORB, or Harris.

    Parameters:
    - image: Input image (color or grayscale)
    - method: 'sift', 'orb', or 'harris' (default='sift')
    - color_mode: 'grayscale' or 'per_channel' (default='grayscale')

    Returns:
    - Keypoints and descriptors (or corner response for Harris)
    """
    if len(image.shape) == 3 and color_mode == 'per_channel':
        image_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)] * 3
    elif len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    if method == 'sift':
        sift = cv2.SIFT_create()
        if color_mode == 'per_channel':
            all_keypoints, all_descriptors = [], []
            for ch in cv2.split(image):
                kp, des = sift.detectAndCompute(ch, None)
                all_keypoints.extend(kp)
                all_descriptors.append(des)
            return all_keypoints, np.vstack(all_descriptors) if len(all_descriptors) > 0 and all(des is not None for des in all_descriptors) else None
        else:
            return sift.detectAndCompute(image_gray, None)
    elif method == 'orb':
        orb = cv2.ORB_create()
        if color_mode == 'per_channel':
            all_keypoints = []
            for ch in cv2.split(image):
                kp, _ = orb.detectAndCompute(ch, None)
                all_keypoints.extend(kp)
            return all_keypoints, None
        else:
            return orb.detectAndCompute(image_gray, None)
    elif method == 'harris':
        if color_mode == 'per_channel':
            channels = cv2.split(image)
            corner_images = [cv2.cornerHarris(
                ch.astype(np.float32), 2, 3, 0.04) for ch in channels]
            combined_corners = np.maximum.reduce(corner_images)
        else:
            combined_corners = cv2.cornerHarris(
                image_gray.astype(np.float32), 2, 3, 0.04)
        return combined_corners, None

# --- Homography Functions ---


def compute_homography(src_points, dst_points):
    """
    Compute homography matrix using RANSAC.
    """
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H


def perspective_to_orthogonal(image, src_points, dst_points):
    """
    Warp perspective image to orthogonal view using four points.
    """
    H = compute_homography(src_points, dst_points)
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, H, (width, height))


def orthogonal_to_perspective(image, src_points, dst_points):
    """
    Warp orthogonal image to perspective view using four points.
    """
    H = compute_homography(src_points, dst_points)
    height, width = image.shape[:2]
    H_inv = np.linalg.inv(H)
    return cv2.warpPerspective(image, H_inv, (width, height))

# --- Harris Corner Detector ---


def apply_harris_corner(image, block_size=2, ksize=3, k=0.04, color_mode='per_channel'):
    """
    Apply Harris Corner Detection with color handling options.
    """
    if len(image.shape) == 3 and color_mode == 'per_channel':
        channels = cv2.split(image)
        corner_images = [cv2.cornerHarris(
            ch.astype(np.float32), block_size, ksize, k) for ch in channels]
        combined_corners = np.maximum.reduce(corner_images)
    else:
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        combined_corners = cv2.cornerHarris(
            image_gray.astype(np.float32), block_size, ksize, k)

    # Make a color copy for marking corners
    if len(image.shape) == 3:
        marked_image = image.copy()
    else:
        marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Mark corners with red dots
    marked_image[combined_corners > 0.01 *
                 combined_corners.max()] = [0, 0, 255]
    return marked_image

# --- ORB Feature Detection ---


def apply_orb_features(image, nfeatures=500, color_mode='per_channel'):
    """
    Apply ORB feature detection with color handling options.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    if len(image.shape) == 3 and color_mode == 'per_channel':
        channels = cv2.split(image)
        all_keypoints = []
        for channel in channels:
            keypoints, _ = orb.detectAndCompute(channel, None)
            all_keypoints.extend(keypoints)
        marked_image = cv2.drawKeypoints(
            image, all_keypoints, None, color=(0, 255, 0), flags=0)
    else:
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        keypoints, _ = orb.detectAndCompute(image_gray, None)
        marked_image = cv2.drawKeypoints(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image,
                                         keypoints, None, color=(0, 255, 0), flags=0)
    return marked_image

# --- Color Space Conversion ---


def rgb_to_hsv(image):
    """
    Convert RGB image to HSV.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


def hsv_to_rgb(image):
    """
    Convert HSV image to RGB.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

# --- Image Segmentation ---


def segment_image(image, method='otsu', color_mode='grayscale'):
    """
    Segment image using Otsu's thresholding or Canny edges.

    Parameters:
    - image: Input image (color or grayscale)
    - method: 'otsu' or 'canny' (default='otsu')
    - color_mode: 'grayscale' or 'per_channel' (default='grayscale')

    Returns:
    - Segmented image (grayscale)
    """
    if len(image.shape) == 3:
        if color_mode == 'per_channel':
            channels = cv2.split(image)
            segmented_channels = []
            for ch in channels:
                if method == 'otsu':
                    _, seg = cv2.threshold(
                        ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif method == 'canny':
                    seg = cv2.Canny(ch, 100, 200)
                segmented_channels.append(seg)
            return cv2.merge(segmented_channels)
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    if method == 'otsu':
        _, segmented = cv2.threshold(
            image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'canny':
        segmented = cv2.Canny(image_gray, 100, 200)
    return segmented

# --- Stereo Stitching ---


def stitch_images(img1, img2):
    """
    Stitch two images using feature matching and homography.
    """
    # Convert images to grayscale for feature detection
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # Detect and compute SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match features
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        # Extract location of good matches
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to align with img2
        height, width = max(
            img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
        stitched = cv2.warpPerspective(img1, H, (width, height))

        # Place img2 in the result image
        stitched[0:img2.shape[0], 0:img2.shape[1]] = img2

        return stitched
    return None

import streamlit as st

# Comprehensive descriptions for all utilities
filter_descriptions = {
    # Basic filters
    "Grayscale": {
        "description": "Converts a color image to grayscale (black and white).",
        "parameters": "None",
        "use_cases": "Preprocessing for many computer vision algorithms, reducing complexity, focusing on contrast.",
        "notes": "Reduces image dimensions from 3 channels to 1 channel."
    },
    "Blur": {
        "description": "Applies Gaussian blur to smooth an image.",
        "parameters": """
        - **kernel_size**: Size of the blur kernel. Larger values create stronger blur effects.
        - **sigma**: Standard deviation of the Gaussian function. Controls the 'spread' of the blur.
        """,
        "use_cases": "Noise reduction, detail smoothing, preprocessing for edge detection.",
        "notes": "Effective for removing high-frequency noise while preserving edges better than simple blur."
    },
    "Edge Detection": {
        "description": "Applies Sobel filter to detect edges in the image by calculating image gradients.",
        "parameters": "**kernel_size**: Size of the Sobel kernel. Affects edge detection precision.",
        "use_cases": "Detecting boundaries, feature extraction, shape analysis.",
        "notes": "Highlights areas of rapid intensity change, which typically correspond to object boundaries."
    },
    "FFT Filter": {
        "description": "Applies a low-pass filter in the frequency domain using Fast Fourier Transform.",
        "parameters": "**cutoff_frequency**: Radius of the circular mask in frequency domain. Lower values remove more high-frequency components.",
        "use_cases": "Noise reduction, pattern analysis, removing periodic noise patterns.",
        "notes": "Works by transforming the image to frequency domain, filtering, and transforming back."
    },
    "Median Filter": {
        "description": "Replaces each pixel with the median value from its surrounding neighborhood.",
        "parameters": "**kernel_size**: Size of the sampling neighborhood.",
        "use_cases": "Salt-and-pepper noise removal, preserving edges while removing noise.",
        "notes": "Better at preserving edges than Gaussian blur while removing noise."
    },
    "Bilateral Filter": {
        "description": "Applies edge-preserving smoothing that reduces noise while preserving edges.",
        "parameters": """
        - **d**: Diameter of each pixel neighborhood.
        - **sigma_color**: Filter sigma in the color space. Larger values mix colors together.
        - **sigma_space**: Filter sigma in the coordinate space. Larger values consider pixels further away.
        """,
        "use_cases": "Noise reduction, cartoonization effects, preprocessing for segmentation.",
        "notes": "Computationally more expensive than simple blurring techniques."
    },
    "Laplacian": {
        "description": "Applies Laplacian operator to detect edges/areas of rapid intensity change in all directions.",
        "parameters": "**kernel_size**: Size of the Laplacian kernel.",
        "use_cases": "Edge detection, feature extraction, sharpening when combined with original image.",
        "notes": "More sensitive to noise than first-order edge detectors like Sobel."
    },
    "Canny Edge": {
        "description": "Advanced edge detection algorithm that applies noise reduction, gradient calculation, and edge tracking.",
        "parameters": """
        - **threshold1**: Lower threshold for the hysteresis procedure.
        - **threshold2**: Upper threshold for the hysteresis procedure.
        """,
        "use_cases": "Precise edge detection, contour finding, feature extraction.",
        "notes": "Generally produces cleaner and more precise edges than simple gradient methods."
    },
    "Morphological": {
        "description": "Applies morphological operations to binary or grayscale images.",
        "parameters": """
        - **operation**: Type of operation (dilate, erode, open, close).
        - **kernel_size**: Size of the structuring element.
        """,
        "use_cases": "Noise removal, connected component analysis, shape manipulation.",
        "notes": "Dilation expands shapes, erosion shrinks them, opening removes small objects, closing fills small holes."
    },
    "Kuwahara Filter": {
        "description": "Non-linear smoothing filter that preserves edges while smoothing regions.",
        "parameters": "**kernel_size**: Size of the filter kernel. Must be odd and ≥5.",
        "use_cases": "Artistic effects, noise reduction while preserving edges, preprocessing for segmentation.",
        "notes": "Creates a painting-like effect by adapting smoothing to local image structure."
    },
    "Cartoon Effect": {
        "description": "Creates a cartoon-like stylization by combining edge detection and color quantization.",
        "parameters": """
        - **bilateral_d**: Diameter for bilateral filtering.
        - **sigma_color**: Color sigma for bilateral filtering.
        - **sigma_space**: Space sigma for bilateral filtering.
        - **edge_threshold1/2**: Thresholds for Canny edge detection.
        """,
        "use_cases": "Artistic effects, image stylization, preprocessing for certain recognition tasks.",
        "notes": "Combines bilateral filtering for smoothing with edge detection for outlines."
    },
    "Pencil Sketch": {
        "description": "Converts an image to a pencil sketch-like drawing.",
        "parameters": "None",
        "use_cases": "Artistic effects, stylization, preprocessing for certain recognition tasks.",
        "notes": "Uses inverted blurring and division to create a sketch-like effect."
    },

    # New features
    "Harris Corner": {
        "description": "Detects corners and interesting points in the image.",
        "parameters": """
        - **block_size**: Size of neighborhood for corner detection.
        - **ksize**: Aperture parameter for the Sobel operator.
        - **k**: Harris detector free parameter.
        - **color_mode**: Whether to apply to each color channel individually or grayscale image.
        """,
        "use_cases": "Feature detection, tracking, image alignment, perspective transforms.",
        "notes": "Marks corners with red dots. Useful for finding interest points that can be tracked between images."
    },
    "ORB Features": {
        "description": "Oriented FAST and Rotated BRIEF feature detector and descriptor.",
        "parameters": """
        - **nfeatures**: Maximum number of features to retain.
        - **color_mode**: Whether to apply to each color channel individually or grayscale image.
        """,
        "use_cases": "Feature matching between images, object recognition, image stitching.",
        "notes": "Faster alternative to SIFT and SURF. Marks features with green circles."
    },
    "HSV Color Space": {
        "description": "Converts image from RGB to HSV color space (Hue, Saturation, Value).",
        "parameters": "None",
        "use_cases": "Color segmentation, color detection, image processing in a more intuitive color space.",
        "notes": "HSV separates color information (hue, saturation) from intensity (value), making it useful for color-based analysis."
    },
    "Perspective to Orthogonal": {
        "description": "Transforms a perspective view into an orthogonal (straight-on) view.",
        "parameters": """
        - **src_points**: Four source points in the original image (corners of the region to transform).
        - **dst_points**: Four destination points defining the transformed rectangle.
        """,
        "use_cases": "Document scanning, whiteboard capture, aerial image correction.",
        "notes": "Used to correct perspective distortion, like making a photograph of a document appear as if taken directly above it."
    },
    "Otsu": {
        "description": "Applies automatic thresholding using Otsu's method to segment an image.",
        "parameters": """
        - **method**: Fixed as 'otsu' for this operation.
        - **color_mode**: Whether to apply to each color channel individually or grayscale image.
        """,
        "use_cases": "Document binarization, object segmentation, foreground-background separation.",
        "notes": "Automatically determines the optimal threshold value by minimizing intra-class variance."
    },
    "Canny Segmentation": {
        "description": "Uses Canny edge detection for image segmentation.",
        "parameters": """
        - **method**: Fixed as 'canny' for this operation.
        - **color_mode**: Whether to apply to each color channel individually or grayscale image.
        """,
        "use_cases": "Object boundary detection, feature extraction, preprocessing for contour analysis.",
        "notes": "Produces binary edge maps that can be used for contour finding and object segmentation."
    },
    "Image Stitching": {
        "description": "Combines multiple images into a single panorama.",
        "parameters": """
        - **img2**: Second image to stitch with the current one.
        """,
        "use_cases": "Creating panoramas, mosaicing, scene reconstruction.",
        "notes": "Requires sufficient overlap and similar exposure between images for good results."
    },

    # Other categories (placeholders for future implementation)
    "Rotate": {
        "description": "Rotates an image by a specified angle.",
        "parameters": "Angle in degrees",
        "use_cases": "Image alignment, orientation correction, data augmentation.",
        "notes": "May introduce interpolation artifacts at certain angles."
    },
    "Flip": {
        "description": "Flips an image horizontally or vertically.",
        "parameters": "Direction (horizontal/vertical)",
        "use_cases": "Data augmentation, mirror images, creating symmetrical compositions.",
        "notes": "Simple operation with no interpolation artifacts."
    },
    "Crop": {
        "description": "Extracts a rectangular region from the image.",
        "parameters": "Coordinates (top, left, width, height)",
        "use_cases": "Focusing on regions of interest, removing unwanted areas, standardizing image sizes.",
        "notes": "Reduces image dimensions."
    },
    # Add more as needed
}


def details():
    with st.expander("Details", expanded=True):
        # Get the currently selected filter/tool
        if "selected_filters" in st.session_state:
            selected = st.session_state.selected_filters.get("filters")

            if selected in filter_descriptions:
                info = filter_descriptions[selected]

                # Display detailed information about the selected filter
                st.markdown(f"## {selected}")
                st.markdown(f"**Description:** {info['description']}")

                if info['parameters'] != "None":
                    st.markdown("**Parameters:**")
                    st.markdown(info['parameters'])

                st.markdown(f"**Use Cases:** {info['use_cases']}")
                st.markdown(f"**Notes:** {info['notes']}")
            else:
                # Default info if the selected filter doesn't have a description yet
                st.write(
                    "This is a computer vision playground that allows you to experiment with different computer vision techniques.")
                st.write("Select a tool from the sidebar to get started.")
                st.markdown("""
                **Categories:**
                - **Filters**: Basic image processing operations like blur, edge detection, etc.
                - **Transformations**: Geometric transformations of images including rotation, perspective transforms, and color spaces.
                - **Segmentation**: Methods to separate an image into meaningful parts.
                - **Detection**: Algorithms to detect specific objects or features in images.
                - **Classification**: Image classification techniques.
                - **Advanced**: More complex operations like image stitching and feature detection.
                """)
        else:
            # Default info if no filter is selected
            st.write(
                "This is a computer vision playground that allows you to experiment with different computer vision techniques.")
            st.write("Select a tool from the sidebar to get started.")
            st.markdown("""
            **Categories:**
            - **Filters**: Basic image processing operations like blur, edge detection, etc.
            - **Transformations**: Geometric transformations of images including rotation, perspective transforms, and color spaces.
            - **Segmentation**: Methods to separate an image into meaningful parts.
            - **Detection**: Algorithms to detect specific objects or features in images.
            - **Classification**: Image classification techniques.
            - **Advanced**: More complex operations like image stitching and feature detection.
            """)

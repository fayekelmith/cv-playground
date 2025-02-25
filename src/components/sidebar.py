import streamlit as st 

interactions : dict = {
    "filters": ["Grayscale", "Blur", "Sharpen", "Edge Detection", "Emboss", "Contour", "Brightness", "Contrast", "Saturation", "Hue", "Gamma", "Thresholding", "Dilation", "Erosion", "Opening", "Closing", "Morphological Gradient", "Top Hat", "Black Hat", "Custom"],
    "transformations": ["Rotate", "Flip", "Crop", "Resize", "Pad", "Warp", "Affine", "Perspective", "Custom"],
    "segmentation": ["Thresholding", "Edge Detection", "Contour", "Watershed", "GrabCut", "Custom"],
    "detection": ["Face", "Object", "Text", "Custom"],
    "classification": ["ImageNet", "Custom"],
}

def sidebar():
    with st.sidebar:
        st.subheader("Tools")
    
        st.markdown("Select a tool to get started.")
    
        for (key, value) in interactions.items():
            st.selectbox(key.capitalize(), value)
    
    
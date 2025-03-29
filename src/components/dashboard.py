from ..services import utils
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

# Add the parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dashboard():
    with st.container():
        st.markdown("<h1>Computer Vision Playground</h1>",
                    unsafe_allow_html=True)

        # Initialize session state for tracking applied filters
        if "applied_filters" not in st.session_state:
            st.session_state.applied_filters = []

        if "original_image" not in st.session_state:
            st.session_state.original_image = None

        if "processed_image" not in st.session_state:
            st.session_state.processed_image = None

        image_section, layers, output_section = st.columns(
            [5, 2, 5], border=True)

        with image_section:
            st.subheader("Input Image")
            uploaded_file = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Convert uploaded file to OpenCV format
                file_bytes = np.asarray(
                    bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.session_state.original_image = image
                st.session_state.processed_image = image.copy()

                # Display the uploaded image
                st.image(image, caption="Uploaded Image",
                         use_column_width=True)

                # Reset applied filters when new image is uploaded
                st.session_state.applied_filters = []
            else:
                # Default image
                st.image("https://picsum.photos/seed/picsum/200/300",
                         caption="Default Image")

        with layers:
            st.subheader("Applied Filters")

            # Display applied filters
            if st.session_state.applied_filters:
                for i, (filter_name, params) in enumerate(st.session_state.applied_filters):
                    st.write(f"{i+1}. {filter_name}")
                    for param_name, param_value in params.items():
                        st.write(f"   - {param_name}: {param_value}")
            else:
                st.write("No filters applied yet")

            # Add button to reset all filters
            if st.button("Reset Filters") and st.session_state.original_image is not None:
                st.session_state.processed_image = st.session_state.original_image.copy()
                st.session_state.applied_filters = []
                st.rerun()

        with output_section:
            st.subheader("Output Image")

            # Process the image if we have an image and selected filters
            if (st.session_state.original_image is not None and
                "selected_filters" in st.session_state and
                    "filter_params" in st.session_state):

                from components.sidebar import filter_functions

                selected_filter = st.session_state.selected_filters.get(
                    "filters")

                # Apply button for the selected filter
                if selected_filter in filter_functions and st.button(f"Apply {selected_filter}"):
                    filter_info = filter_functions[selected_filter]
                    filter_params = st.session_state.filter_params.get(
                        selected_filter, {})

                    # Apply the filter
                    processed = filter_info["function"](
                        st.session_state.processed_image,
                        **filter_params
                    )

                    # Update the processed image
                    st.session_state.processed_image = processed

                    # Add to applied filters list
                    st.session_state.applied_filters.append(
                        (selected_filter, filter_params))

                    # Rerun to update the UI
                    st.rerun()

            # Display the processed image or a default image
            if st.session_state.processed_image is not None:
                st.image(
                    st.session_state.processed_image,
                    caption="Processed Image",
                    use_column_width=True
                )
            else:
                st.image("https://picsum.photos/id/237/200/300",
                         caption="Default Output")

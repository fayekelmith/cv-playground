import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

from src.services import utils
from src.components.sidebar import filter_functions


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
                         use_container_width=True)

                # Reset applied filters when new image is uploaded
                st.session_state.applied_filters = []
            else:
                st.info("Please upload an image to get started")

        with layers:
            st.subheader("Applied Filters")
            # Display applied filters
            if st.session_state.applied_filters:
                # Use st.expander for better display if list gets long
                with st.expander("Show Applied Filters", expanded=True):
                    for i, (filter_name, params) in enumerate(st.session_state.applied_filters):
                        st.write(f"{i+1}. {filter_name}")
                        # Display parameters concisely
                        param_str = ", ".join(
                            [f"{k}={v}" for k, v in params.items()])
                        if param_str:
                            st.caption(f"   Params: {param_str}")
            else:
                st.write("No filters applied yet.")

            # Add button to reset all filters
            if st.button("Reset Filters", key="reset_filters_button") and st.session_state.original_image is not None:
                st.session_state.processed_image = st.session_state.original_image.copy()
                st.session_state.applied_filters = []
                st.rerun()

        with output_section:
            st.subheader("Output Image")

            # Process the image if we have an image and selected filters
            if st.session_state.original_image is not None:
                if "selected_filters" in st.session_state:
                    selected_filter = st.session_state.selected_filters.get(
                        "filters")

                    if selected_filter in filter_functions:
                        filter_info = filter_functions[selected_filter]

                        if "filter_params" in st.session_state and selected_filter in st.session_state.filter_params:
                            filter_params = st.session_state.filter_params.get(
                                selected_filter, {})

                            button_key = f"apply_button_{selected_filter}"
                            if st.button(f"Apply {selected_filter}", key=button_key):
                                try:

                                    input_image = st.session_state.processed_image.copy()

                                    processed = filter_info["function"](
                                        input_image,
                                        **filter_params
                                    )

                                    if processed is None:
                                        raise ValueError(
                                            f"Filter returned None")

                                    processed = np.asarray(
                                        processed, dtype=np.uint8)

                                    if len(processed.shape) == 2 and len(input_image.shape) == 3:
                                        processed = cv2.cvtColor(
                                            processed, cv2.COLOR_GRAY2RGB)

                                    st.session_state.processed_image = processed

                                    current_filter_entry = (
                                        selected_filter, filter_params.copy())
                                    st.session_state.applied_filters.append(
                                        current_filter_entry)

                                    st.success(
                                        f"Applied {selected_filter} filter successfully")

                                except Exception as e:
                                    st.error(
                                        f"Error applying filter: {str(e)}")
                                    import traceback
                                    st.sidebar.write(traceback.format_exc())
                        else:
                            # This might occur if the filter has no parameters defined in sidebar.py
                            st.warning(
                                f"Parameters configuration not found for {selected_filter}. Applying without params if possible.")
                            # Attempt to apply without params if button pressed
                            button_key = f"apply_button_{selected_filter}_no_params"
                            if st.button(f"Apply {selected_filter}", key=button_key):
                                try:
                                    # ... (similar try block as above, but without **filter_params) ...
                                    input_image = st.session_state.processed_image.copy()
                                    processed = filter_info["function"](
                                        input_image)  # Call without params

                                    # ... (rest of the processing, state update, and rerun logic) ...
                                    if processed is None:
                                        raise ValueError(
                                            f"Filter returned None")
                                    processed = np.asarray(
                                        processed, dtype=np.uint8)
                                    if len(processed.shape) == 2 and len(input_image.shape) == 3:
                                        processed = cv2.cvtColor(
                                            processed, cv2.COLOR_GRAY2RGB)

                                    st.session_state.processed_image = processed
                                    st.session_state.applied_filters.append(
                                        (selected_filter, {}))  # Append with empty params
                                    st.success(
                                        f"Applied {selected_filter} filter successfully (no params).")

                                except Exception as e:
                                    st.error(
                                        f"Error applying filter (no params): {str(e)}")
                                    import traceback
                                    st.sidebar.write(traceback.format_exc())

                else:
                    st.info("Select a filter from the sidebar.")

            # Display the processed image or a default image
            if st.session_state.processed_image is not None:
                processed_image_display = np.asarray(
                    st.session_state.processed_image, dtype=np.uint8)
                st.image(
                    processed_image_display,
                    caption=f"Processed Image ({processed_image_display.shape})",
                    use_container_width=True,
                )
            else:
                st.info("Upload an image and apply filters to see results.")

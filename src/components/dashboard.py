import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

from src.services import utils
from src.components.sidebar import tool_functions


def dashboard():
    with st.container():
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 20px;'>CV Playground - Theory Sucks! </h1>", unsafe_allow_html=True)

        if "applied_filters" not in st.session_state:
            st.session_state.applied_filters = []

        if "original_image" not in st.session_state:
            st.session_state.original_image = None

        if "processed_image" not in st.session_state:
            st.session_state.processed_image = None

        image_section, output_section = st.columns(
            [5, 5], border=False)

        with image_section:
            st.markdown(
                "<h2 style='text-align: center; margin-bottom: 20px;'>Input Image </h2>", unsafe_allow_html=True)
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

        with output_section:
            st.markdown(
                "<h2 style='text-align: center; margin-bottom: 20px;'>Output Image</h2>", unsafe_allow_html=True)

            # Process the image if we have an image and selected filters
            if st.session_state.original_image is not None:
                # Check if we have an active tool selected from any category
                if "active_tool" in st.session_state and st.session_state.active_tool:
                    active_tool = st.session_state.active_tool
                    active_category = st.session_state.active_category

                    if active_tool in tool_functions:
                        st.markdown(f"""
                        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;">
                            <b>Selected Tool:</b> {active_tool} <span style="color: #666;">({active_category.capitalize()})</span>
                        </div>
                        """, unsafe_allow_html=True)

                        tool_info = tool_functions[active_tool]

                        # Handle special cases for tools needing custom UI
                        if active_tool == "Perspective to Orthogonal":
                            st.write("Select four corners in the image:")

                            # Create an interface for selecting corners
                            h, w = st.session_state.processed_image.shape[:2]

                            # Create 2 columns for corner selection
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("Top-Left")
                                tl_x = st.slider("X", 0, w, w//4, key="tl_x")
                                tl_y = st.slider("Y", 0, h, h//4, key="tl_y")

                                st.write("Bottom-Left")
                                bl_x = st.slider("X", 0, w, w//4, key="bl_x")
                                bl_y = st.slider("Y", 0, h, 3*h//4, key="bl_y")

                            with col2:
                                st.write("Top-Right")
                                tr_x = st.slider("X", 0, w, 3*w//4, key="tr_x")
                                tr_y = st.slider("Y", 0, h, h//4, key="tr_y")

                                st.write("Bottom-Right")
                                br_x = st.slider("X", 0, w, 3*w//4, key="br_x")
                                br_y = st.slider("Y", 0, h, 3*h//4, key="br_y")

                            # Create the source and destination point arrays
                            src_points = np.float32(
                                [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]])
                            dst_points = np.float32(
                                [[0, 0], [w, 0], [w, h], [0, h]])

                            # Apply perspective transform when button is clicked
                            if st.button("Apply Perspective Transform"):
                                try:
                                    input_image = st.session_state.processed_image.copy()
                                    processed = utils.perspective_to_orthogonal(
                                        input_image, src_points, dst_points)

                                    if processed is None:
                                        raise ValueError(
                                            "Transform returned None")

                                    processed = np.asarray(
                                        processed, dtype=np.uint8)
                                    st.session_state.processed_image = processed
                                    st.session_state.applied_filters.append(
                                        (active_tool, {"src_points": src_points.tolist(
                                        ), "dst_points": dst_points.tolist()})
                                    )
                                    st.success(
                                        f"Applied {active_tool} successfully")
                                except Exception as e:
                                    st.error(
                                        f"Error applying transform: {str(e)}")
                                    import traceback
                                    st.sidebar.write(traceback.format_exc())

                        elif active_tool == "Image Stitching":
                            st.write(
                                "Upload a second image to stitch with the current image")

                            second_image_file = st.file_uploader(
                                "Upload second image", type=["jpg", "jpeg", "png"], key="second_image_uploader"
                            )

                            if second_image_file is not None:
                                # Convert uploaded file to OpenCV format
                                second_file_bytes = np.asarray(
                                    bytearray(second_image_file.read()), dtype=np.uint8)
                                second_image = cv2.imdecode(
                                    second_file_bytes, cv2.IMREAD_COLOR)

                                # Convert BGR to RGB
                                second_image = cv2.cvtColor(
                                    second_image, cv2.COLOR_BGR2RGB)

                                # Show the second image
                                st.image(
                                    second_image, caption="Second Image", use_container_width=True)

                                # Apply stitching when button is clicked
                                if st.button("Apply Image Stitching"):
                                    try:
                                        input_image = st.session_state.processed_image.copy()
                                        processed = utils.stitch_images(
                                            input_image, second_image)

                                        if processed is None:
                                            st.error(
                                                "Could not stitch images - not enough matching features found.")
                                        else:
                                            processed = np.asarray(
                                                processed, dtype=np.uint8)
                                            st.session_state.processed_image = processed
                                            st.session_state.applied_filters.append(
                                                (active_tool, {}))
                                            st.success(
                                                f"Applied {active_tool} successfully")
                                    except Exception as e:
                                        st.error(
                                            f"Error applying image stitching: {str(e)}")
                                        import traceback
                                        st.sidebar.write(
                                            traceback.format_exc())

                        # Standard parameter-based tools
                        elif "filter_params" in st.session_state and active_tool in st.session_state.filter_params:
                            tool_params = st.session_state.filter_params.get(
                                active_tool, {})

                            button_key = f"apply_button_{active_tool}"
                            if st.button(f"Apply {active_tool}", key=button_key):
                                try:
                                    input_image = st.session_state.processed_image.copy()

                                    processed = tool_info["function"](
                                        input_image,
                                        **tool_params
                                    )

                                    if processed is None:
                                        raise ValueError(f"Tool returned None")

                                    processed = np.asarray(
                                        processed, dtype=np.uint8)

                                    if len(processed.shape) == 2 and len(input_image.shape) == 3:
                                        processed = cv2.cvtColor(
                                            processed, cv2.COLOR_GRAY2RGB)

                                    st.session_state.processed_image = processed

                                    current_filter_entry = (
                                        active_tool, tool_params.copy())
                                    st.session_state.applied_filters.append(
                                        current_filter_entry)

                                    st.success(
                                        f"Applied {active_tool} successfully")

                                except Exception as e:
                                    st.error(
                                        f"Error applying {active_tool}: {str(e)}")
                                    import traceback
                                    st.sidebar.write(traceback.format_exc())
                        else:
                            # Tools with no parameters
                            button_key = f"apply_button_{active_tool}_no_params"
                            if st.button(f"Apply {active_tool}", key=button_key):
                                try:
                                    input_image = st.session_state.processed_image.copy()
                                    processed = tool_info["function"](
                                        input_image)

                                    if processed is None:
                                        raise ValueError(
                                            f"{active_tool} returned None")

                                    processed = np.asarray(
                                        processed, dtype=np.uint8)

                                    if len(processed.shape) == 2 and len(input_image.shape) == 3:
                                        processed = cv2.cvtColor(
                                            processed, cv2.COLOR_GRAY2RGB)

                                    st.session_state.processed_image = processed
                                    st.session_state.applied_filters.append(
                                        (active_tool, {}))
                                    st.success(
                                        f"Applied {active_tool} successfully")

                                except Exception as e:
                                    st.error(
                                        f"Error applying {active_tool}: {str(e)}")
                                    import traceback
                                    st.sidebar.write(traceback.format_exc())
                else:
                    st.info("Select a tool from the sidebar")

            # Display the processed image or a default image
            if st.session_state.processed_image is not None:
                processed_image_display = np.asarray(
                    st.session_state.processed_image, dtype=np.uint8)
                st.image(
                    processed_image_display,
                    caption=f"Processed Image ({processed_image_display.shape})",
                    use_container_width=True,
                )

                # Display applied filters
                if st.session_state.applied_filters:
                    with st.expander("Applied Filters"):
                        for i, (filter_name, params) in enumerate(st.session_state.applied_filters):
                            st.markdown(f"**{i+1}. {filter_name}**")
                            if params:
                                param_text = ", ".join([f"{k}: {v}" for k, v in params.items()
                                                        if k not in ["src_points", "dst_points"]])  # Skip complex params
                                if param_text:
                                    st.caption(f"Parameters: {param_text}")
            else:
                st.info("Upload an image and apply filters to see results.")

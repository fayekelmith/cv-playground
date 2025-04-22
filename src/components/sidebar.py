from ..services import utils
import streamlit as st
import sys
import os

# Add the parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

interactions: dict = {
    "filters": ["Grayscale", "Blur", "Edge Detection", "FFT Filter", "Median Filter",
                "Bilateral Filter", "Laplacian", "Canny Edge", "Morphological",
                "Kuwahara Filter", "Cartoon Effect", "Pencil Sketch", "Custom"],
    "transformations": ["Rotate", "Flip", "Crop", "Resize", "Pad", "Warp",
                        "Perspective to Orthogonal", "HSV Color Space", "Custom"],
    "segmentation": ["Thresholding", "Edge Detection", "Contour", "Otsu", "Canny Segmentation",
                     "Watershed", "GrabCut", "Custom"],
    "detection": ["Face", "Harris Corner", "ORB Features", "Object", "Text", "Custom"],
    "classification": ["ImageNet", "Custom"],
    "advanced": ["Image Stitching", "Homography", "Feature Detection"]
}

# Dictionary mapping tool names to functions
tool_functions = {
    # Filters
    "Blur": {
        "function": utils.apply_gaussian_filter,
        "params": {
            "kernel_size": (1, 31, 2, 5),
            "sigma": (0.1, 10.0, 0.1, 1.0)
        }
    },
    "Edge Detection": {
        "function": utils.apply_sobel_filter,
        "params": {
            "kernel_size": (1, 31, 2, 3)
        }
    },
    "FFT Filter": {
        "function": utils.apply_fft_filter,
        "params": {
            "cutoff_frequency": (1, 100, 1, 30)
        }
    },
    "Median Filter": {
        "function": utils.apply_median_filter,
        "params": {
            "kernel_size": (1, 31, 2, 3)
        }
    },
    "Bilateral Filter": {
        "function": utils.apply_bilateral_filter,
        "params": {
            "d": (1, 15, 2, 9),
            "sigma_color": (10, 150, 5, 75),
            "sigma_space": (10, 150, 5, 75)
        }
    },
    "Laplacian": {
        "function": utils.apply_laplacian_filter,
        "params": {
            "kernel_size": (1, 31, 2, 3)
        }
    },
    "Canny Edge": {
        "function": utils.apply_canny_filter,
        "params": {
            "threshold1": (0, 300, 5, 100),
            "threshold2": (0, 300, 5, 200)
        }
    },
    "Morphological": {
        "function": utils.apply_morphological_operation,
        "params": {
            "operation": ["dilate", "erode", "open", "close"],
            "kernel_size": (1, 21, 2, 5)
        }
    },
    "Kuwahara Filter": {
        "function": utils.apply_kuwahara_filter,
        "params": {
            "kernel_size": (5, 25, 2, 5)
        }
    },
    "Cartoon Effect": {
        "function": utils.apply_cartoon_filter,
        "params": {
            "bilateral_d": (1, 15, 2, 9),
            "sigma_color": (10, 150, 5, 75),
            "sigma_space": (10, 150, 5, 75),
            "edge_threshold1": (0, 300, 5, 100),
            "edge_threshold2": (0, 300, 5, 200)
        }
    },
    "Pencil Sketch": {
        "function": utils.apply_pencil_sketch,
        "params": {}
    },

    # Detection tools
    "Harris Corner": {
        "function": utils.apply_harris_corner,
        "params": {
            "block_size": (1, 10, 1, 2),
            "ksize": (1, 31, 2, 3),
            "k": (0.01, 0.1, 0.01, 0.04),
            "color_mode": ["grayscale", "per_channel"]
        }
    },
    "ORB Features": {
        "function": utils.apply_orb_features,
        "params": {
            "nfeatures": (100, 1000, 100, 500),
            "color_mode": ["grayscale", "per_channel"]
        }
    },

    # Transformation tools
    "HSV Color Space": {
        "function": utils.rgb_to_hsv,
        "params": {}
    },
    "Perspective to Orthogonal": {
        "function": utils.perspective_to_orthogonal,
        "params": {
            "src_points": "corners",
            "dst_points": "corners"
        }
    },

    # Segmentation tools
    "Otsu": {
        "function": utils.segment_image,
        "params": {
            "method": ["otsu"],
            "color_mode": ["grayscale", "per_channel"]
        }
    },
    "Canny Segmentation": {
        "function": utils.segment_image,
        "params": {
            "method": ["canny"],
            "color_mode": ["grayscale", "per_channel"]
        }
    },

    # Advanced tools
    "Image Stitching": {
        "function": utils.stitch_images,
        "params": {
            "img2": "second_image"
        }
    }
}

# For backward compatibility
filter_functions = tool_functions


def sidebar():
    with st.sidebar:
        st.subheader("Tools")
        st.markdown("Select a tool to get started.")

        selected_filters = {}
        active_category = None
        active_tool = None

        # First, scan through all categories to display them
        for (key, value) in interactions.items():
            selection = st.selectbox(
                key.capitalize(), value, key=f"select_{key}")
            selected_filters[key] = selection

            # Find the first non-Custom selection to determine the active category and tool
            if selection != "Custom" and selection in tool_functions and active_category is None:
                active_category = key
                active_tool = selection

        st.markdown("---")

        # Display parameters for the active tool (regardless of category)
        if active_tool and active_tool in tool_functions:
            st.markdown(f"### {active_tool} Parameters")
            st.markdown(f"*Category: {active_category.capitalize()}*")

            tool_info = tool_functions[active_tool]
            tool_params = {}

            # Create controls for each parameter
            for param_name, param_range in tool_info["params"].items():
                # Skip special parameters that need custom UI elements
                if param_range == "corners" or param_range == "second_image":
                    continue

                # If param_range is a list, create a dropdown
                if isinstance(param_range, list):
                    tool_params[param_name] = st.selectbox(
                        f"{param_name.capitalize()}",
                        options=param_range,
                        key=f"param_{active_tool}_{param_name}"
                    )
                # Otherwise create a slider
                else:
                    min_val, max_val, step, default = param_range
                    tool_params[param_name] = st.slider(
                        f"{param_name.capitalize()}",
                        min_value=min_val,
                        max_value=max_val,
                        step=step,
                        value=default,
                        key=f"param_{active_tool}_{param_name}"
                    )

            # Store the parameters in session state
            if "filter_params" not in st.session_state:
                st.session_state.filter_params = {}
            st.session_state.filter_params[active_tool] = tool_params

            # Add an info section indicating the active tool
            st.markdown("---")
            st.info(
                f"Active tool: {active_tool} ({active_category.capitalize()})")

        # Store selected filters in session state
        st.session_state.selected_filters = selected_filters
        st.session_state.active_category = active_category
        st.session_state.active_tool = active_tool

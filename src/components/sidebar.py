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
    "transformations": ["Rotate", "Flip", "Crop", "Resize", "Pad", "Warp", "Affine", "Perspective", "Custom"],
    "segmentation": ["Thresholding", "Edge Detection", "Contour", "Watershed", "GrabCut", "Custom"],
    "detection": ["Face", "Object", "Text", "Custom"],
    "classification": ["ImageNet", "Custom"],
}

# Dictionary mapping filter names to functions
filter_functions = {
    "Blur": {
        "function": utils.apply_gaussian_filter,
        "params": {
            "kernel_size": (1, 31, 2, 5),  # (min, max, step, default)
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
    }
}


def sidebar():
    with st.sidebar:
        st.subheader("Tools")

        st.markdown("Select a tool to get started.")

        selected_filters = {}
        for (key, value) in interactions.items():
            selection = st.selectbox(
                key.capitalize(), value, key=f"select_{key}")
            selected_filters[key] = selection

            # If a filter is selected and it exists in our mapping, show parameter controls
            if key == "filters" and selection in filter_functions:
                filter_info = filter_functions[selection]
                filter_params = {}

               # Create controls for each parameter
                for param_name, param_range in filter_info["params"].items():
                    # If param_range is a list, create a dropdown
                    if isinstance(param_range, list):
                        filter_params[param_name] = st.selectbox(
                            f"{param_name.capitalize()}",
                            options=param_range,
                            key=f"param_{selection}_{param_name}"
                        )
                    # Otherwise create a slider
                    else:
                        min_val, max_val, step, default = param_range
                        filter_params[param_name] = st.slider(
                            f"{param_name.capitalize()}",
                            min_value=min_val,
                            max_value=max_val,
                            step=step,
                            value=default,
                            key=f"param_{selection}_{param_name}"
                        )
                # Store the parameters in session state
                if "filter_params" not in st.session_state:
                    st.session_state.filter_params = {}
                st.session_state.filter_params[selection] = filter_params

        # Store selected filters in session state
        st.session_state.selected_filters = selected_filters

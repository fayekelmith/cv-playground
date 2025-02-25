import streamlit as st 

def dashboard():
    with st.container():
        st.markdown("<h1>Computer Vision Playground</h1>",unsafe_allow_html=True)
        
       
        image_section, layers, output_section = st.columns([5,2,5], border=True)
        with image_section:
            st.image("https://picsum.photos/seed/picsum/200/300")
            
        with layers:
            st.write("Layers")
        
        with output_section:
            st.image("https://picsum.photos/id/237/200/300")
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Test basic functionality first
st.set_page_config(page_title="CMB Detection", layout="wide")

# Basic CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #D36BA3, #974578);
    color: white;
}
.text-white {
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "CMB Detection",
        ["Home", "Test", "Debug"],
        default_index=0
    )

if selected == "Home":
    st.title("🧠 CMB Detection System")
    st.markdown('<div class="text-white">Basic app is working!</div>', unsafe_allow_html=True)
    
    # Test basic functionality
    if st.button("Test Button"):
        st.success("✅ Button works!")
    
    # Test numpy
    try:
        arr = np.array([1, 2, 3])
        st.write(f"Numpy test: {arr}")
    except Exception as e:
        st.error(f"Numpy error: {e}")
    
    # Test matplotlib
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Matplotlib error: {e}")

elif selected == "Test":
    st.title("🧪 Test Page")
    
    # Test requests
    try:
        st.info("Testing HF connection...")
        response = requests.get("https://huggingface.co/datasets/anbndct/NII_MRI_CMB", timeout=10)
        if response.status_code == 200:
            st.success("✅ HF connection works!")
        else:
            st.warning(f"HF response: {response.status_code}")
    except Exception as e:
        st.error(f"HF connection error: {e}")
    
    # Test file upload
    uploaded_file = st.file_uploader("Test file upload", type=['npy', 'txt'])
    if uploaded_file:
        st.success("✅ File upload works!")

elif selected == "Debug":
    st.title("🔧 Debug Info")
    
    # Show environment info
    import sys
    st.write("Python version:", sys.version)
    
    # Show installed packages
    try:
        import tensorflow as tf
        st.write("TensorFlow version:", tf.__version__)
    except Exception as e:
        st.error(f"TensorFlow error: {e}")
    
    try:
        import nibabel as nib
        st.write("Nibabel imported successfully")
    except Exception as e:
        st.error(f"Nibabel error: {e}")
    
    try:
        import scipy
        st.write("Scipy version:", scipy.__version__)
    except Exception as e:
        st.error(f"Scipy error: {e}")

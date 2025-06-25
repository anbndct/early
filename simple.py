import streamlit as st

# Ultra minimal test - no external packages
st.set_page_config(page_title="CMB Detection Test", layout="wide")

st.title("🧠 CMB Detection System - Debug Mode")

st.write("If you can see this, basic Streamlit is working!")

# Test basic Streamlit functionality
if st.button("Test Button"):
    st.success("✅ Button works!")

# Simple sidebar without option_menu
with st.sidebar:
    page = st.selectbox("Select Page", ["Home", "Test", "Debug"])

if page == "Home":
    st.header("🏠 Home")
    st.write("Welcome to CMB Detection System")
    
    # Test basic Python
    try:
        import numpy as np
        st.success("✅ NumPy works!")
        arr = np.array([1, 2, 3])
        st.write(f"NumPy array: {arr}")
    except Exception as e:
        st.error(f"❌ NumPy error: {e}")
    
    try:
        import pandas as pd
        st.success("✅ Pandas works!")
    except Exception as e:
        st.error(f"❌ Pandas error: {e}")

elif page == "Test":
    st.header("🧪 Test Page")
    
    # Test matplotlib
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        st.pyplot(fig)
        st.success("✅ Matplotlib works!")
    except Exception as e:
        st.error(f"❌ Matplotlib error: {e}")
    
    # Test requests
    try:
        import requests
        st.success("✅ Requests imported!")
    except Exception as e:
        st.error(f"❌ Requests error: {e}")

elif page == "Debug":
    st.header("🔧 Debug Info")
    
    import sys
    st.write("**Python Version:**", sys.version)
    
    # Check problematic imports one by one
    imports_to_test = [
        "numpy",
        "pandas", 
        "matplotlib",
        "requests",
        "streamlit_option_menu",
        "scipy",
        "tensorflow",
        "nibabel",
        "sklearn"
    ]
    
    for pkg in imports_to_test:
        try:
            __import__(pkg)
            st.success(f"✅ {pkg} - OK")
        except ImportError as e:
            st.error(f"❌ {pkg} - Import Error: {e}")
        except Exception as e:
            st.error(f"❌ {pkg} - Other Error: {e}")

# Simple CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #D36BA3, #974578);
    color: white;
}
</style>
""", unsafe_allow_html=True)

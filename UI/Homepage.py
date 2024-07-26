import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Oral Lesion App", 
    page_icon="oral.png",  
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        color: #FF6F61;
        text-align: left;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #6A1B9A;
        text-align: left;
        margin-bottom: 40px;
    }
    .banner {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subheader
st.markdown('<h1 class="title">Welcome to the Oral Lesion App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Tools for Training Models and Object Detection</p>', unsafe_allow_html=True)

# Banner image
st.image("coverpage.png", use_column_width=False, output_format="auto", width=600)

# Introductory text
st.write("""
Welcome to the Oral Lesion App! This platform provides state-of-the-art tools for training machine learning models and performing object detection tasks. Explore our features to enhance your workflow and get accurate results.
""")

# Getting started section
st.markdown("""
### Get Started
**Training Models**: Learn how to train models efficiently using our easy-to-follow guides.
**Object Detection Tools**: Utilize our robust tools to detect and classify oral lesions .

""")

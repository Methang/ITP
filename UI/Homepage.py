import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="Oral Lesion App", 
    page_icon="😏", 
    layout="wide"
)

# Add a title with styling
st.title("Welcome to the Oral Lesion App")
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        color: #FF6F61;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #6A1B9A;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display a subtitle
st.markdown('<p class="subheader">Tools for Training Models and Object Detection</p>', unsafe_allow_html=True)

# Add a welcome image or graphic
st.image("coverpage.png", use_column_width=True)  # Replace with your own image URL

# Add a description with Markdown
st.write("""
Welcome to the Oral Lesion App! This platform provides state-of-the-art tools for training machine learning models and performing object detection tasks. Explore our features to enhance your workflow and get accurate results.
""")

# Add a section for getting started
st.markdown("""
### Get Started
1. **Training Models**: Learn how to train models efficiently using our easy-to-follow guides.
2. **Object Detection Tools**: Utilize our robust tools to detect and classify oral lesions with high precision.
3. **Support**: Visit our support section for help and FAQs.
""")

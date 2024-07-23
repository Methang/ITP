import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from PIL import Image  # Import the Image module from PIL
import os
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
st.set_page_config(
    page_title="Oral Lesion App", page_icon="😏")

st.title("Object Detection Page")

try:

    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union

    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""


def main():

     # Add radio buttons for selection
    detection_method = st.radio("Choose detection Model:", ("MobileNet-v2", "YOLOv8", "GroundingDINO"))
    # Add radio buttons for selection
    training_method = st.radio("Select what you want to Run the Model with:", ("CPU (Default)", "GPU (will be used if selected and available)"))

    # Show the selected detection method
    # st.write("You selected:", detection_method)
    # st.info(__doc__)
    # st.markdown(STYLE, unsafe_allow_html=True)
    # Get file path input from user
    data_path = st.text_input("Enter path of image:", key="data_path")

    if data_path:
        if os.path.exists(data_path):
            st.write("Image found!")
            st.image(data_path, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            # Proceed with further actions
            if st.button("Run"):
                if detection_method == "GroundingDINO":
                    print("")
                    #CALL GROUNDING
                    # Set device to CPU
                    if torch.cuda.is_available() and training_method == "GPU (will be used if selected and available)":
                        device = torch.device('cuda')
                        st.write("using gpu")
                    else:
                        device = torch.device("cpu")
                        st.write("using cpu")
                    model = load_model("groundingdino/config/tuning.py", "groundingdino_swint_ogc.pth")
                    model = model.to(device)
                    IMAGE_PATH = data_path
                    TEXT_PROMPT = "lesion"

                    BOX_THRESHOLD = 0.35
                    TEXT_THRESHOLD = 0.25

                    image_source, image = load_image(IMAGE_PATH)

                    # Move the image to CPU
                    image = image.to(device)

                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )

                    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                    st.image("testnolesion_1.jpg")
                print("")
        else:
            st.error("Invalid file path. Please provide a valid path.")

    # file = st.file_uploader("Upload file", type=["csv", "png", "jpg"])
    # show_file = st.empty()

    # if not file:
    #     show_file.info("Please upload a file of type: " +
    #                    ", ".join(["csv", "png", "jpg"]))
    #     return

    # content = file.getvalue()

    # if isinstance(file, BytesIO):
    #     show_file.image(file)
    #     if st.button("Start Training"):
    #         print("")
    # else:
    #     data = pd.read_csv(file)
    #     st.dataframe(data.head(10))
    # file.close()


   

main()

import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from PIL import Image
import os
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
<<<<<<< HEAD
from ultralytics import YOLO
=======
from fastai.vision.all import load_learner, PILImage
from fastai.vision import *
from fastai.vision.all import *
from fastai.learner import *
>>>>>>> fastai

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
<<<<<<< HEAD

    # Add radio buttons for selection
    detection_method = st.radio("Choose detection Model:", ("MobileNet-v2", "YOLOv8", "GroundingDINO"))
=======
    modelpath =''
     # Add radio buttons for selection
    detection_method = st.radio("Choose detection Model:", ("Fast.AI", "YOLOv8", "GroundingDINO"))
    if detection_method == "Fast.AI":
        modelpath = st.text_input("Enter path of Exported Model:", key=modelpath)
        if modelpath:
            if os.path.exists(modelpath):
                st.write("Valid Model Path!")
>>>>>>> fastai
    # Add radio buttons for selection
    training_method = st.radio("Select what you want to Run the Model with:", ("CPU (Default)", "GPU (will be used if selected and available)"))

    # Get file path input from user
    data_path = st.text_input("Enter path of image:", key="data_path")

    if data_path:
        if os.path.exists(data_path):
            st.write("Image found!")
            st.image(data_path, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

            # Proceed with further actions
            if st.button("Run"):
<<<<<<< HEAD
              
                if detection_method == "YOLOv8":
                    # Set device to CPU or GPU
=======
                if detection_method == "Fast.AI":
                    # Load the exported model
                    modelpath2 = Path(modelpath)
                    learn_inf = load_learner(modelpath2)

                    # Path to the new image you want to predict
                    img_path = Path(data_path)

                    # Open the image
                    img = PILImage.create(img_path)

                    # Make a prediction
                    pred_class, pred_idx, outputs = learn_inf.predict(img)
                    print(f"Predicted class: {pred_class}")

                if detection_method == "GroundingDINO":
                    print("")
                    #CALL GROUNDING
                    # Set device to CPU
>>>>>>> fastai
                    if torch.cuda.is_available() and training_method == "GPU (will be used if selected and available)":
                        device = torch.device('cuda')
                        st.write("Using GPU")
                    else:
                        device = torch.device("cpu")
                        st.write("Using CPU")
                    
                    model = YOLO('best.pt')  # Load the trained YOLOv8 model
                    results = model(data_path)  # Perform inference
                    annotated_img = results[0].plot()  # Annotate image with detections
                   
                    st.image(annotated_img, caption='Detected Image')  # Display the annotated image
                
                elif detection_method == "GroundingDINO":
                    
                    if torch.cuda.is_available() and training_method == "GPU (will be used if selected and available)":
                        device = torch.device('cuda')
                        st.write("Using GPU")
                    else:
                        device = torch.device("cpu")
                        st.write("Using CPU")
                    model = load_model("groundingdino/config/tuning.py", "groundingdino_swint_ogc.pth")
                    model = model.to(device)
                    IMAGE_PATH = data_path
                    TEXT_PROMPT = "lesion"
                    BOX_THRESHOLD = 0.35
                    TEXT_THRESHOLD = 0.25
                    image_source, image = load_image(IMAGE_PATH)
                    image = image.to(device)
                    boxes, logits, phrases = predict(
                        model=model,
                        image=image,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )
<<<<<<< HEAD
                    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                    st.image(annotated_frame)
=======

                    annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                    st.image("testnolesion_1.jpg")
                print("")
>>>>>>> fastai
        else:
            st.error("Invalid file path. Please provide a valid path.")

main()

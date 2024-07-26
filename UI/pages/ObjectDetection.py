import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from PIL import Image
import os
import torch
from torchvision import transforms, models
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from fastai.vision.all import load_learner, PILImage
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(page_title="Oral Lesion App", page_icon="oral.png")

st.title("Object Detection Page")

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

def main():
    modelpath = ''
    
    # Add radio buttons for selection
    detection_method = st.radio("Choose detection Model:", ("Fast.AI", "YOLOv8", "GroundingDINO"))
    
    if detection_method == "Fast.AI":
        modelpath = st.text_input("Enter path of Exported Model:", key="modelpath")
        model_exists = modelpath and os.path.exists(modelpath)
        if model_exists:
            st.write("Valid Model Path!")
        else:
            st.write("Please enter a valid model path.")
    
    training_method = st.radio("Select what you want to Run the Model with:", ("CPU (Default)", "GPU (will be used if selected and available)"))

    data_path = st.text_input("Enter path of image:", key="data_path")

    # Button to run model
    if st.button("Run"):
        if data_path and os.path.exists(data_path):
            st.write("Image found!")
            st.image(data_path, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            
            if detection_method == "Fast.AI" and model_exists:
                modelpath2 = Path(modelpath)
                learn_inf = load_learner(modelpath2)
                img_path = Path(data_path)
                img = PILImage.create(img_path)
                pred_class, pred_idx, outputs = learn_inf.predict(img)
                st.write(f"Predicted class: {pred_class}")
            
            elif detection_method == "YOLOv8":
                device = torch.device('cuda' if torch.cuda.is_available() and training_method == "GPU (will be used if selected and available)" else 'cpu')
                st.write("Using GPU" if device.type == 'cuda' else "Using CPU")
                model = YOLO('best.pt')
                results = model(data_path)
                annotated_img = results[0].plot()
                st.image(annotated_img, caption='Detected Image')
            
            elif detection_method == "GroundingDINO":
                device = torch.device('cuda' if torch.cuda.is_available() and training_method == "GPU (will be used if selected and available)" else 'cpu')
                st.write("Using GPU" if device.type == 'cuda' else "Using CPU")
                model = load_model("groundingdino/config/tuning.py", "groundingdino_swint_ogc.pth")
                model = model.to(device)
                IMAGE_PATH = data_path
                TEXT_PROMPT = "lesion"
                BOX_THRESHOLD = 0.35
                TEXT_THRESHOLD = 0.25
                image_source, image = load_image(IMAGE_PATH)
                image = image.to(device)
                boxes, logits, phrases = predict(model=model, image=image, caption=TEXT_PROMPT, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                st.image(annotated_frame)
        else:
            st.error("Invalid file path. Please provide a valid path.")

main()

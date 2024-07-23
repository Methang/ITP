import streamlit as st
import pandas as pd
import os


# Use st.cache_data instead of deprecated st.cache
@st.cache_data
def load_data():
    try:
        # Load your dataset here
        # For example, assuming a CSV file named 'your_data.csv'
        data = pd.read_csv('your_data.csv')
        return data
    except FileNotFoundError:
        #st.error("Data file not found. Please make sure the file 'your_data.csv' is available.")
        return None

# Function to simulate training and evaluation
def train_and_evaluate_model(data, target_column, param1, param2):
    # Add your model training and evaluation logic here
    # For now, let's just print the parameters as an example
    print(f"Training model with parameters: param1={param1}, param2={param2}")
    



# Model Parameters on the main page
st.header('Model Training')


# Get file path input from user
data_path = st.text_input("Enter data file path:", key="data_path")

if data_path:
    if os.path.exists(data_path):
        st.write("Path exists")
        # Proceed with further actions
    else:
        st.error("Invalid file path. Please provide a valid path.")

# Get file path input from user
output_path = st.text_input("Enter output file path:", key="output_path")

if output_path:
    if os.path.exists(output_path):
        st.write("Path exists")
        # Proceed with further actions
    else:
        st.error("Invalid file path. Please provide a valid path.")



# Sample parameter inputs (replace these with your actual parameters)
    
#param1 = st.slider('Parameter 1', min_value=1, max_value=100, value=50)
param1 = st.radio('Models', ['MobileNet-v2', 'YOLOv8'])


# if using YOLOv8
if param1=="YOLOv8":
    epoch = 1
    batch_size = 1
    learning_rate = 1
    epoch = st.number_input('Insert epoch', min_value=1, value=1, step=1)
    batch_size = st.number_input('Insert batch size', min_value=1, value=1, step=1)
    learning_rate = st.number_input('Insert a learning rate', min_value=1, value=1, step=1)

if param1=="MobileNet-v2":
    epoch = 1
    batch_size = 1
    epoch = st.number_input('Insert epoch', min_value=1, value=1, step=1)
    batch_size = st.number_input('Insert batch size', min_value=1, value=1, step=1)
    #learningRate = st.number_input('Insert a learning rate', min_value=1, value=1, step=1)

target_column=0

# Add radio buttons for selection
training_method = st.radio("Select what you want to Train the Model with:", ("CPU (Default)", "GPU (will be used if selected and available)"))


# Check if data loading was successful before proceeding
data = load_data()
if data is not None:
    target_column = 'target_column'  # Replace 'target_column' with your actual target variable

# Display the "Start Training" button
if st.button("Start Training"):
    # Trigger training when the button is clicked
    print("")
if st.button("Export trained model"):
    # Trigger training when the button is clicked
    train_and_evaluate_model(data, target_column, epoch, batch_size)
import streamlit as st
import random
from PIL import Image
import cv2 
import os
import torch

# Define a list of test images (replace with your actual image paths)

# Specify the path to the folder containing your image files
folder_path = "/DATA/pal14/M22MA002/EasyVqa/easy_vqa123/data/test"
# test_images =
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # Loop through the image files and process them
for image_file in image_files:
        # Create the full file path
        file_path = os.path.join(folder_path, image_file)

        # Read the image using OpenCV
        test_images = cv2.imread(file_path)

        if test_images is not None:
            # Process the image here
            # For example, you can apply image processing operations using OpenCV

            # Display the image (for demonstration purposes)
            # cv2.imshow('Image', test_images)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Failed to read {image_file}")






# Define a list of sample questions
sample_questions = [
    "What color is the object?",
    "What is the shape of the object?",
    "Where is the object located?",
    "How many objects are there?",
]
import json
json_file_path="/DATA/pal14/M22MA002/EasyVqa/easy_vqa123/data/test/questions.json"
with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        for i in range(0,len(data)):
            questions = data[i][0]
            # print(questions)
# Function to load and display a random image from the test set
def display_random_image(folder_path):
    # Get a list of image files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    if not image_files:
        st.write("No image files found in the specified folder.")
        return

    # Choose a random image from the list
    random_image_filename = random.choice(image_files)

    # Create the full path to the random image
    random_image_path = os.path.join(folder_path, random_image_filename)

    # Display the random image using Streamlit
    image = Image.open(random_image_path)
    st.image(image, caption="Random Test Image", use_column_width=True)

# Specify the path to your image folder
folder_path = folder_path

# Call the function to display a random image
display_random_image(folder_path)
    # st.image(image, caption="Random Test Image", use_column_width=True)

# Function to generate a random question
def generate_random_question():
    random_question = random.choice(sample_questions)
    return random_question

# Load and initialize your model (replace with actual model loading code)
def load_model1():
    # Load and return your pre-trained model
    # map_location=torch.device('cpu')

    model4 = torch.load("/DATA/pal14/M22MA002/EasyVqa/easyvqa_finetuned_epoch_10.model")
    model.load_state_dict(model4)
    torch.load()
    return model
import torch

def load_model():
    # Load and return your pre-trained model
    map_location = torch.device('cpu')

    # Load the model into the variable model4
    model4 = torch.load("C:\\Users\\hp\\OneDrive\\Desktop\\MTP\\EasyVQA\\easy_vqa123\\easyvqa_finetuned_epoch_10.model", map_location)
    
    return model4  # Return the loaded model

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def vqa_model(question, image_path):
    # Load a pre-trained VQA model
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Tokenize the question and context (image) text
    inputs = tokenizer(question, image_path, return_tensors="pt", padding=True, truncation=True)

    # Get the model's output
    with torch.no_grad():
        output = model(**inputs)

    # Extract the answer from the model's output
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1]))

    return answer

# Example usage
# question = "What is in the image?"
# image_path = "path_to_your_image.jpg"
# answer = vqa_model(question, image_path)
# print("Answer:", answer)

# Perform model inference
# def vqa_model(model, image, question):
#     # Replace this with your actual model inference code
#     # This is a placeholder implementation
#     return "This is a placeholder for the model's answer."

# Streamlit UI
st.title("Visual Question Answering App")

st.markdown(
    """
    <style>
    .main-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.2);
    }
    .header-text {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .image-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 4px rgba(0, 0, 0, 0.1);
    }
    .question-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 4px rgba(0, 0, 0, 0.1);
    }
    .button-container {
        margin-top: 20px;
    }
    .model-answer {
        font-size: 18px;
        font-weight: bold;
        color: #555;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Container
with st.container() as main_container:
    st.markdown('<p class="header-text">Visual Question Answering App</p>', unsafe_allow_html=True)

    # Random Test Image Section
    with st.container() as image_container:
        st.markdown('<p class="header-text">Random Test Image</p>', unsafe_allow_html=True)
        display_random_image(folder_path)
    # Get a list of image files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # if not image_files:
    #     st.write("No image files found in the specified folder.")
    #     return

    # Choose a random image from the list
    random_image_filename = random.choice(image_files)

    # Create the full path to the random image
    random_image_path = os.path.join(folder_path, random_image_filename)

    # Display the random image using Streamlit
    image = Image.open(random_image_path)
    st.image(image, caption="Random Test Image", use_column_width=True)

# Specify the path to your image folder
# folder_path = "path_to_your_image_folder"

# Call the function to display a random image
display_random_image(folder_path)
    # Random Question Section
with st.container() as question_container:
    st.markdown('<p class="header-text">Random Question</p>', unsafe_allow_html=True)
    input_question = st.text_area("Ask a question:", generate_random_question())

# Button Container
with st.container() as button_container:
    st.markdown('<p class="button-container">', unsafe_allow_html=True)
    if st.button("Change Random Image"):
        selected_image = random.choice(test_images)
        display_random_image(folder_path)
    if st.button("Generate Random Question"):
        input_question = generate_random_question()
        st.text("Random Question: " + input_question)

# Load the model
model = load_model()

# Model Answer Section (replace with actual model prediction)
st.markdown('<p class="model-answer">Model Answer</p>', unsafe_allow_html=True)
if st.button("Get Model Answer"):
    selected_image = random.choice(test_images)
    input_question = generate_random_question()
    # Perform model inference by passing the displayed image and entered question directly
    model_answer = vqa_model( selected_image, input_question)  # Implement your model's inference function

    # Display the model's answer
    st.write(model_answer)

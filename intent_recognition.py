import streamlit as st
import pickle
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import zipfile
import os
import pickle  # Assuming your model is saved as a pickle file

# Load tokenizer (update based on the model you trained with)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to load the classifier from a zip file
def load_model_from_zip(zip_file_path, extracted_path="/"):
    """
    Extract the .zip file and load the model
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    
    # Assuming the model is saved as a pickle file in the extracted folder
    model_path = "/content/distilbert_intent_model"  # Update the file path if different
    
    # Load the model and tokenizer explicitly
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create the pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )
    
    return classifier

# Function to load the label encoder from a pickle file
def load_label_encoder(encoder_file_path):
    with open(encoder_file_path, "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return label_encoder

# Intent classification function
def classify_intent(text, classifier, label_encoder):
    # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # embeddings = model(**inputs).last_hidden_state[:, 0, :].detach().numpy()  # CLS token embeddings
    prediction = classifier(text)
    intent = label_encoder.inverse_transform([int(prediction[0]['label'].replace('LABEL_', ''))])
    return intent

# Streamlit app
def main():
    st.title("Intent Recognition App")
    
    # Upload trained model (.zip)
    st.header("Upload Trained Model and Label Encoder")
    model_file = st.file_uploader("Upload the ZIP file containing the trained model", type="zip")
    encoder_file = st.file_uploader("Upload the PKL file for the label encoder", type="pkl")
    
    if model_file and encoder_file:
        with st.spinner("Loading model and encoder..."):
            # Save uploaded files locally
            zip_file_path = "distilbert_intent_model.zip"
            encoder_file_path = "label_encoder.pkl"
            
            with open(zip_file_path, "wb") as file:
                file.write(model_file.getbuffer())
            
            with open(encoder_file_path, "wb") as file:
                file.write(encoder_file.getbuffer())
            
            # Load the model and encoder
            classifier = load_model_from_zip(zip_file_path)
            label_encoder = load_label_encoder(encoder_file_path)
            # model = AutoModel.from_pretrained("bert-base-uncased")
            st.success("Model and Label Encoder loaded successfully!")
    
    # Input text for intent classification
    st.header("Input Text")
    user_input = st.text_area("Enter a sentence to classify its intent:", height=100)

    # Predict intent
    if st.button("Classify Intent"):
        if user_input.strip() and model_file and encoder_file:
            intent = classify_intent(user_input, classifier, label_encoder)
            st.write(f"Predicted Intent: **{intent}**")
        else:
            st.error("Please upload the model, encoder, and enter valid text!")

if __name__ == "__main__":
    main()

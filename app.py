# Import necessary libraries
import joblib
import numpy as np
import warnings
from PIL import Image
from rake_nltk import Rake
import streamlit as st
import nltk
import pandas as pd

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('wordnet')

# Import text analysis functions
from text_analysis import clean_text

# Warnings ignore 
warnings.filterwarnings(action='ignore')

# Describing the Web Application 
st.title('TextAnalyzeMaster')

# Display image
try:
    display = Image.open('images/display.jpg')
    display = np.array(display)
    st.image(display)
except FileNotFoundError:
    st.error("Error: Image file not found.")

# Sentiment Analysis Tool
st.header("Sentiment Analysis Tool")
st.subheader("Enter the statement that you want to analyze")

# Provide a unique key for the text area widget
text_input = st.text_area("Enter sentence", height=50, key='text_input_area')

# Model Selection 
model_select = st.selectbox("Model Selection", ["Naive Bayes", "Logistic Regression"], key='model_select')

if st.button("Predict", key='predict_button'):
    try:
        # Clean the text input
        cleaned_text = clean_text(text_input)
        
        # Load the model 
        if model_select == "Logistic Regression":
            sentiment_model = joblib.load('Models/logistic_regression_sentiment_model.sav')
        elif model_select == "Naive Bayes":
            sentiment_model = joblib.load('Models/naive_bayes_sentiment_model.sav')
        
        # Load the vectorizer
        vectorizer = joblib.load('Models/tfidf_vectorizer_sentiment_model.sav')

        # Vectorize the cleaned inputs 
        vec_inputs = vectorizer.transform([cleaned_text])

        # Keyword extraction 
        r = Rake(language='english')
        r.extract_keywords_from_text(cleaned_text)
        
        # Get the important phrases
        phrases = r.get_ranked_phrases()

        # Make the prediction 
        prediction = sentiment_model.predict(vec_inputs)
        if prediction[0] == 1:
            st.write("This statement is **Positive**")
        else:
            st.write("This statement is **Negative**")

        # Display the important phrases
        st.write("These are the **keywords** causing the above sentiment:")
        for i, p in enumerate(phrases):
            st.write(f"{i + 1}. {p}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

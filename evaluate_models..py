#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate models with the saved vectorizer.

Created on Sun Aug  4 20:40:15 2024

@author: jayanth
"""

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    # Provide the absolute path to your dataset
    file_path = '/Users/jayanth/TextAnalyzeMaster/data/sentiment140.csv'
    try:
        df = pd.read_csv(file_path, encoding='latin1', header=None)
        print(f"Loaded with encoding: latin1")
        print("Columns in the dataset:", df.columns)
        print("First few rows of the dataset:")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    return df

def preprocess_data(df):
    # Relabel positive samples
    df[0] = df[0].replace(4, 1)

    # Manually specify column indices
    text_column_index = 5  # Index of the column for text
    label_column_index = 0  # Index of the column for labels

    # Ensure the expected columns are in the dataset
    if text_column_index not in df.columns or label_column_index not in df.columns:
        raise ValueError(f"Error: Missing expected columns '{text_column_index}' or '{label_column_index}'")
    
    # Extract features and labels
    X = df[text_column_index].astype(str)  
    y = df[label_column_index].astype(int)  

    return X, y

def evaluate_models():
    # Load data
    df = load_data()
    X, y = preprocess_data(df)

    # Load the vectorizer
    vectorizer = joblib.load('Models/tfidf_vectorizer_sentiment_model.sav')

    # Transform the input data
    X_vectorized = vectorizer.transform(X)

    # Load models
    model_paths = {
        "Naive Bayes": 'Models/naive_bayes_sentiment_model.sav',
        "Logistic Regression": 'Models/logistic_regression_sentiment_model.sav'
    }

    # Evaluate each model
    for model_name, model_path in model_paths.items():
        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_vectorized)
            accuracy = accuracy_score(y, y_pred)
            print(f"Accuracy of {model_name}: {accuracy:.2f}")
            
            # Confusion Matrix and Classification Report
            print(f"Confusion Matrix for {model_name}:")
            print(confusion_matrix(y, y_pred))
            
            print(f"Classification Report for {model_name}:")
            print(classification_report(y, y_pred))
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

if __name__ == "__main__":
    evaluate_models()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:40:15 2024

@author: jayanth
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

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

def check_data_distribution(y):
    # Check for class imbalance in the dataset
    positive_count = sum(y == 1)
    negative_count = sum(y == 0)
    print(f"Positive samples: {positive_count}")
    print(f"Negative samples: {negative_count}")

def balance_data(X, y):
    # Use SMOTE to balance the training data
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_and_save_models():
    # Load data
    df = load_data()
    
    X, y = preprocess_data(df)

    # Check data distribution
    check_data_distribution(y)

    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize vectorizer with a fixed number of features
    vectorizer = TfidfVectorizer(max_features=500000)

    # Fit and transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Balance the training data using SMOTE
    X_train_vectorized, y_train = balance_data(X_train_vectorized, y_train)

    # Transform test data
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize models with optimized settings
    models = {
        "Naive Bayes": BernoulliNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1)
    }

    # Train, evaluate, and save each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_vectorized, y_train)

        # Save the trained model
        joblib.dump(model, f'Models/{model_name.lower().replace(" ", "_")}_sentiment_model.sav')

        # Evaluate model
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {model_name}: {accuracy:.2f}")

        # Confusion Matrix and Classification Report
        print(f"Confusion Matrix for {model_name}:")
        print(confusion_matrix(y_test, y_pred))
        
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

    # Save the vectorizer
    joblib.dump(vectorizer, 'Models/tfidf_vectorizer_sentiment_model.sav')

    # Fine-tuning the Logistic Regression Model
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['saga']
    }

    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, n_jobs=-1), param_grid, cv=5)
    grid_search.fit(X_train_vectorized, y_train)
    print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    # Save the best logistic regression model
    best_lr_model = grid_search.best_estimator_
    joblib.dump(best_lr_model, 'Models/best_logistic_regression_sentiment_model.sav')

if __name__ == "__main__":
    train_and_save_models()

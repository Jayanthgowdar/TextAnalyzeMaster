
# TextAnalyzeMaster

**TextAnalyzeMaster** is a comprehensive text analytics project that includes various applications like sentiment analysis, text classification, topic modeling, and spam filtering. The project utilizes several natural language processing (NLP) techniques and machine learning models to analyze and derive meaningful insights from text data.

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results and Findings](#results-and-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This project aims to explore various text analysis techniques using Python. It includes different modules to preprocess data, train models, evaluate them, and deploy a simple application for user interaction. The models and techniques applied can be used for sentiment analysis, spam filtering, topic modeling, and text classification.

## Motivation

With the exponential growth of textual data, extracting valuable information from this data is crucial for decision-making processes across various industries. This project was motivated by the need to automate the process of understanding and categorizing large amounts of text data efficiently.

## Dataset

The project utilizes the **Twitter Sentiment Dataset** by Saurabh Shahane, available on Kaggle. This dataset includes a large number of tweets labeled with sentiments, making it suitable for training and evaluating sentiment analysis models.

## Methods

The project involves the following key steps:

1. **Data Preprocessing**:
    - Tokenization, stopwords removal, and lemmatization using NLTK and SpaCy.
    - Text cleaning and normalization.

2. **Model Training**:
    - Implementing machine learning models like Logistic Regression, Naive Bayes, and LSTM using the preprocessed data.
    - Training models using Python libraries like scikit-learn and TensorFlow.

3. **Evaluation**:
    - Evaluating models based on accuracy, precision, recall, and F1 score.
    - Cross-validation to ensure model robustness.

4. **Deployment**:
    - Deploying the application using Streamlit, allowing users to input text and receive analysis results in real-time.

## Results and Findings

The project successfully demonstrates the effectiveness of various machine learning techniques in text analysis. The LSTM model, in particular, showed promising results for sentiment analysis, outperforming traditional models in accuracy and F1 score. The spam filter was also able to categorize messages with high precision.

## Project Structure

The project consists of the following files:

- `train_model.py`: Script to train and save the machine learning models.
- `preprocess_data.py`: Script for preprocessing the text data.
- `evaluate_models.py`: Script for evaluating the performance of the trained models.
- `app.py`: Streamlit app for user interaction.
- `text_analysis.py`: Core functions for text analysis including sentiment analysis, topic modeling, and spam filtering.
- `requirements.txt`: List of Python libraries and dependencies required for the project.
- `setup.sh`: Shell script for setting up the Streamlit configuration.

## Installation

To run this project, follow these steps:

1. Ensure you have Python installed on your machine.

2. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up Streamlit configuration by running:

    ```bash
    sh setup.sh
    ```

## Usage

After installing the dependencies, you can start the Streamlit app using:

```bash
streamlit run app.py
```

This command will launch a local server where you can interact with the app. The app will allow you to input text and receive real-time analysis results.



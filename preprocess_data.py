#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:27:43 2024

@author: jayanth
"""

import pandas as pd
from text_analysis import clean_text

# Define paths for the input and output datasets
DATASET_PATH = '/Users/jayanth/TextAnalyzeMaster/data/sentiment140.csv'
CLEANED_DATASET_PATH = '/Users/jayanth/TextAnalyzeMaster/data/cleaned_sentiment140.csv'

# Load the dataset
dataset = pd.read_csv(DATASET_PATH, encoding='latin1')
dataset.columns = ['label', 'id', 'date', 'query', 'user', 'text']

# Apply text cleaning to the 'text' column
dataset['cleaned_text'] = dataset['text'].apply(clean_text)

# Save the cleaned dataset
dataset.to_csv(CLEANED_DATASET_PATH, index=False)

print(f"Cleaned dataset saved to {CLEANED_DATASET_PATH}")

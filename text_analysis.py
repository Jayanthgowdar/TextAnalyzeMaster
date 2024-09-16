# text_analysis.py

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Constants
STOPWORDS = stopwords.words('english')
STOPWORDS += ['said']

# Text cleaning function
def clean_text(text):
    '''
    Function which returns a clean text 
    '''    
    # Lower case 
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d', '', text)
    
    # Remove newlines and extra spaces
    text = re.sub(r'\n', '', text).strip()
    
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove Stopwords and Lemmatize the data
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text)
    
    return text

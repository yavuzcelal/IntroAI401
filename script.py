# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

# download the punkt tokenizer model (for requirements 3 & 4)
nltk.download('punkt')

# Loadingthe data from the provided CSV file
data = pd.read_csv('KatyPerry.csv')

# Display the first few rows of the dataframe to understand its structure
data.head()

# Basic data exploration

# Count of spam vs non-spam comments
spam_count = data['CLASS'].value_counts()

# Sample spam comments
spam_comments_sample = data[data['CLASS'] == 1]['CONTENT'].sample(5, random_state=1)

# Sample non-spam comments
non_spam_comments_sample = data[data['CLASS'] == 0]['CONTENT'].sample(5, random_state=1)

# General statistics
total_comments = len(data)
unique_authors = data['AUTHOR'].nunique()
comments_with_date = data['DATE'].notnull().sum()

spam_count, spam_comments_sample, non_spam_comments_sample, total_comments, unique_authors, comments_with_date

# requirement 3
# NLTK preprocessing function
def nltk_preprocess(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# applying NLTK preprocessing
preprocessed_content = data['CONTENT'].apply(nltk_preprocess)

# initializing and applying countvectorizer
vectorizer = CountVectorizer()
content_bow_nltk = vectorizer.fit_transform(preprocessed_content)

# requirement 4
# highlights of countvectorizer output
# printing the shape of countvectorizer output and a handful of feature names from countvectorizer
print("Data Shape (Bag-of-Words):", content_bow_nltk.shape)
print("Feature Names Sample (Bag-of-Words):", vectorizer.get_feature_names_out()[:10])


# requirement 5 STARTED BUT INCOMPLETE
# applying tf-idf transformation
tfidf_transformer = TfidfTransformer()
content_tfidf = tfidf_transformer.fit_transform(content_bow_nltk)

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

# download the punkt tokenizer model (for requirements 3 & 4)
nltk.download('punkt')

# Loadingthe data from the provided CSV file
path='C:/Users/kesha/OneDrive/Desktop/Semester1/Introduction to AI/project/IntroAI401'
filename = 'KatyPerry.csv'
fullpath = os.path.join(path,filename)
data = pd.read_csv(fullpath)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

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


# requirement 5 
# applying tf-idf transformation
tfidf_transformer = TfidfTransformer()
content_tfidf = tfidf_transformer.fit_transform(content_bow_nltk)

# Displaying the shape of the TF-IDF matrix
print("Data Shape (TF-IDF):", content_tfidf.shape)

# Print a sample TF-IDF
print("(TF-IDF):", content_tfidf[:10])

# requirement 6
# Shuffle the dataset
df_shuffled_data = data.sample(frac=1, random_state=42)
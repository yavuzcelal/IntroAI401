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
from sklearn.naive_bayes import MultinomialNB


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

# requirement 7
# Splitting the dataset into 75% training and 25% testing
train_size = int(0.75 * len(df_shuffled_data))
train_data = df_shuffled_data[:train_size]
test_data = df_shuffled_data[train_size:]

# Separate the class from the features
X_train_raw = train_data['CONTENT']
y_train = train_data['CLASS']
X_test_raw = test_data['CONTENT']
y_test = test_data['CLASS']

# Applying NLTK preprocessing and CountVectorizer to training and testing sets
X_train = X_train_raw.apply(nltk_preprocess)
X_test = X_test_raw.apply(nltk_preprocess)
X_train_tfidf = tfidf_transformer.fit_transform(vectorizer.fit_transform(X_train))
X_test_tfidf = tfidf_transformer.transform(vectorizer.transform(X_test))

# requirement 8
# Initialize and fit the Naive Bayes classifier

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# requirement 9
# Cross validate the model on the training data using 5-fold and print the mean results of model accuracy
scores = cross_val_score(nb_classifier, X_train_tfidf, y_train, cv=5)
print("Mean accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# requirement 10
# Test the model on the test data, print the confusion matrix and the accuracy of the model
predictions = nb_classifier.predict(X_test_tfidf)
conf_matrix = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy_score(y_test, predictions))

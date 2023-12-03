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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


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

print(f'spam_count:{spam_count}, spam_comments_sample:{spam_comments_sample},non_spam_comments_sample:{non_spam_comments_sample}, total_comments:{total_comments}, unique_authors{unique_authors}, comments_with_date:{comments_with_date}')

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

# Step 9: Cross-validation on the training data
# Assuming 'nb_classifier' is your trained Naive Bayes classifier

# Convert the TF-IDF matrix to an array for cross-validation
X_train_array = X_train_tfidf.toarray()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(nb_classifier, X_train_array, y_train, cv=5)

# Print the mean accuracy across folds
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Step 10: Evaluate the model on the test data

# Predictions on the test set
y_pred = nb_classifier.predict(X_test_tfidf)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)


# Example comments
comments = [
    "I really enjoyed reading this article. It provided valuable insights and information.",
    "The product quality is excellent, and the customer service is top-notch.",
    "This song has been on repeat for days. Can't get enough of it!",
    "The weather today is perfect for a relaxing outdoor picnic with friends.",
    "Click here for a chance to win a free iPhone! Limited time offer!",
    "Make thousands of dollars from home with this easy money-making method. Join now!",
]

# Preprocess the comments
preprocessed_comments = [nltk_preprocess(comment) for comment in comments]

# Vectorize and transform using CountVectorizer and TF-IDF transformer
comments_tfidf = tfidf_transformer.transform(vectorizer.transform(preprocessed_comments))

# Make predictions
predictions = nb_classifier.predict(comments_tfidf)

# Display the results
for comment, prediction in zip(comments, predictions):
    print(f"Comment: {comment}")
    print(f"Predicted Class: {'Spam' if prediction == 1 else 'Non-Spam'}")
    print("\n" + "-"*40 + "\n")




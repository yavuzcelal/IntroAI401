# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Load the data from the provided CSV file
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
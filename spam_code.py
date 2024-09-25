##BUSINESS PROBLEM:
'''Spam mail, or junk mail, is a type of email
that is sent to a massive number of users at one time, frequently containing cryptic
messages, scams, or most dangerously, phishing content.
In this Project, use Python to build an email spam detector. Then, use machine learning to
train the spam detector to recognize and classify emails into spam and non-spam. '''

##BUSINESS OBJECTIVE: 
'''Detection of spam mail by using machine learning models.'''
##BUSINESS CONSTRAINTS:
'''Minimize the manuall spam detection techniques.'''

##BUSINESS SUCCESS CRITERIA:
'''1.Economic success criteria:Increasing the profit by 20% by reducing customer churn.
2.Business success criteria: Reducing the user churn.
3.ML success criteria:Achieve accuracy of the model by 89%.'''

##Data dictionary :
'''v1 : level
v2: messages'''

########################################################################################################################

##Import necessary libraries
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Documents\Data science\project2\TASK4\spam.csv", encoding='latin-1')
df

# Explore the dataset
print(df.head())

# Data cleaning: Remove unnecessary columns and rename
df = df[['v1', 'v2']]  # Keep only relevant columns
df.columns = ['label', 'message']  # Rename columns for clarity

# Check for missing values
df.isnull().sum()

# Label encoding: Convert 'spam' to 1 and 'ham' to 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Data preprocessing: Removing punctuation and converting to lower case
def preprocess_text(text):
    text = text.lower()  # Convert text to lower case
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

df['message_clean'] = df['message'].apply(preprocess_text)

# Split data into training and testing sets
X = df['message_clean']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Save the model and the TF-IDF vectorizer for later use
import joblib
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')





















































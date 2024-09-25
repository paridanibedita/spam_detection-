import streamlit as st
import string 
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_detector_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Title for the app
st.title('Email Spam Detector')

# Input for the email text
email_text = st.text_area("Enter the email content here:")

# Button for prediction
if st.button('Predict'):
    # Preprocess the input email
    email_processed = email_text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Convert the input email into numerical features using TF-IDF
    email_tfidf = tfidf.transform([email_processed])
    
    # Predict if it's spam or not
    prediction = model.predict(email_tfidf)
    
    # Show the prediction result
    if prediction == 1:
        st.error('This email is SPAM.')
    else:
        st.success('This email is NOT SPAM.')


import streamlit as st
import pickle

# Load model + vectorizer
model, vectorizer = pickle.load(open("spam_model.pkl", "rb"))

st.title("📩 Spam Email/SMS Classifier")
st.write("Enter a message and the model will predict if it is Spam or Not Spam.")

# Input text
user_input = st.text_area("Message", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Transform input
        input_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0]

        # Show result
        if prediction == 1:
            st.error(f"🚨 This Email is SPAM! (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ This Email is Not SPAM. (Confidence: {prob[0]*100:.2f}%)")

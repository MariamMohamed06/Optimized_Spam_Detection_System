import streamlit as st
import joblib

# Load the final trained model and vectorizer
model = joblib.load("models/final_spam_model.pkl")
vectorizer = joblib.load("models/final_count_vectorizer.pkl")

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Optimized Spam Detection System")
st.write("Enter a message below to check whether it is Spam or Ham.")
# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won $1000. Click here to claim your prize."
)

# Prediction button
if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Transform the message using the saved vectorizer
        message_features = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_features)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(message_features)[0]

        ham_probability = probabilities[0]
        spam_probability = probabilities[1]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write(f"Spam Confidence: **{spam_probability:.2%}**")
        else:
            st.success("✅ HAM (LEGITIMATE MESSAGE)")
            st.write(f"Ham Confidence: **{ham_probability:.2%}**")

        # Show probabilities
        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Ham", f"{ham_probability:.2%}")

        with col2:
            st.metric("Spam", f"{spam_probability:.2%}")

st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "98.84%")

with col2:
    st.metric("Precision", "97.60%")

with col3:
    st.metric("Recall", "93.13%")

with col4:
    st.metric("F1-Score", "95.31%")
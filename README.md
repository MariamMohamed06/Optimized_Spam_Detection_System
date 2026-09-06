# 📩 Optimized Spam Detection System

A Machine Learning system for detecting whether an SMS message is **Spam** or **Ham (Legitimate)**.

The project uses traditional Machine Learning techniques with **Scikit-learn** and provides a simple **Streamlit** web application for real-time predictions.

---

## 🎯 Project Objective

The main objective is to build a reliable spam detection system by:

- Preprocessing SMS messages.
- Converting text into numerical features.
- Training multiple Machine Learning classifiers.
- Evaluating models using standard classification metrics.
- Performing hyperparameter and feature tuning.
- Selecting the best-performing model.
- Deploying the final model using Streamlit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- Git & GitHub

---

## 📊 Dataset

The project uses an SMS spam dataset containing two main classes:

- **Ham** — Legitimate messages.
- **Spam** — Unwanted or fraudulent messages.

The cleaned dataset is stored in:

```text
data/cleaned_spam.csv
🤖 Machine Learning Models

Several traditional Machine Learning approaches were evaluated using:

Feature Extraction
CountVectorizer
TF-IDF Vectorizer
Classifiers
Multinomial Naive Bayes
Logistic Regression
LinearSVC

The models were compared using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
⚙️ Optimization

Hyperparameter and feature tuning were performed using training data only.

The experiments included:

MultinomialNB alpha
CountVectorizer ngram_range
min_df
max_df
LinearSVC C
LinearSVC class_weight

The tuning experiments did not outperform the original best-performing configuration on the held-out test set, so the original configuration was retained as the final model.

🏆 Final Model

The selected final model is:

CountVectorizer + Multinomial Naive Bayes

with:

alpha = 0.5
Final Test Performance
Metric	Score
Accuracy	98.84%
Precision	97.60%
Recall	93.13%
F1-Score	95.31%
Confusion Matrix
[[900, 3],
 [  9, 122]]

The final evaluation was performed on an untouched test set to provide an honest estimate of model performance.

🌐 Streamlit Application

The project includes a Streamlit web application that allows users to enter an SMS message and receive:

Spam/Ham prediction
Prediction confidence
Ham probability
Spam probability
Final model performance metrics
🚀 How to Run
1. Clone the repository
git clone https://github.com/MariamMohamed06/Optimized_Spam_Detection_System.git
2. Navigate to the project directory
cd Optimized_Spam_Detection_System
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit application
python -m streamlit run app.py

The application will open in your browser.

📁 Project Structure
Optimized_Spam_Detection_System/
│
├── data/
│   ├── cleaned_spam.csv
│   └── spam.csv
│
├── models/
│   ├── final_count_vectorizer.pkl
│   ├── final_spam_model.pkl
│   ├── count_vectorizer.pkl
│   ├── countvectorizer_linearsvc.pkl
│   ├── countvectorizer_logisticregression.pkl
│   ├── countvectorizer_multinomialnb.pkl
│   ├── tf_idf_linearsvc.pkl
│   ├── tf_idf_logisticregression.pkl
│   ├── tf_idf_multinomialnb.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── Data Preprocessing.ipynb
│   ├── Models.ipynb
│   ├── Evaluation.ipynb
│   ├── Optimization.ipynb
│   └── evaluate.py
│
├── outputs/
│   ├── confusion_matrices.png
│   ├── f1_ranking.png
│   ├── metrics_comparison.png
│   ├── misclassified_examples.csv
│   └── model_comparison_results.csv
│
├── app.py
├── requirements.txt
└── README.md
👥 Project Workflow

The project workflow consists of:

Data Collection & Preprocessing
Feature Extraction
Model Training
Model Evaluation
Hyperparameter & Feature Optimization
Final Model Selection
Streamlit Deployment
📌 Conclusion

The final system achieved an F1-Score of 95.31% and 98.84% accuracy on the held-out test set.

The project demonstrates a complete traditional Machine Learning pipeline for SMS spam detection, from preprocessing and model comparison to optimization and deployment.
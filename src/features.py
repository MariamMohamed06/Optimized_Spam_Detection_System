import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42

def load_data(filepath="../data/cleaned_spam.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["clean_message", "label"]).reset_index(drop=True)
    
    X = df["clean_message"].astype(str)
    y = df["label"].map({"ham": 0, "spam": 1})
    return X, y

def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def extract_and_save_features(X_train, X_test, output_dir="../models/"):
    # CountVectorizer
    count_vec = CountVectorizer()
    X_train_count = count_vec.fit_transform(X_train)
    X_test_count = count_vec.transform(X_test)
    joblib.dump(count_vec, f"{output_dir}count_vectorizer.pkl")

    # TF-IDF Vectorizer
    tfidf_vec = TfidfVectorizer()
    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_test_tfidf = tfidf_vec.transform(X_test)
    joblib.dump(tfidf_vec, f"{output_dir}tfidf_vectorizer.pkl")

    feature_sets = {
        "CountVectorizer": (X_train_count, X_test_count),
        "TF-IDF": (X_train_tfidf, X_test_tfidf),
    }

    return feature_sets

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_sets = extract_and_save_features(X_train, X_test)
    print("Feature extraction complete and vectorizers saved.")
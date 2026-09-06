import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from features import load_data, split_data, extract_and_save_features

RANDOM_STATE = 42

def get_models(random_state=RANDOM_STATE):
    return {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LinearSVC": LinearSVC(
            C=1.0, random_state=random_state, class_weight="balanced", max_iter=5000
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
    }

def check_fit(model, X_train, y_train, X_test, y_test):
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    gap = train_accuracy - test_accuracy

    print(f"Train Accuracy: {round(train_accuracy, 4)}")
    print(f"Test Accuracy: {round(test_accuracy, 4)}")
    print(f"Train-Test Gap: {round(gap, 4)}")

    if train_accuracy < 0.85:
        print("Status: Underfitting")
    elif gap > 0.05:
        print("Status: Possible Overfitting")
    else:
        print("Status: Good Fit")

def train_and_evaluate(feature_sets, y_train, y_test, models_dir="../models/"):
    trained_models = {}

    for feature_name, (X_train_feat, X_test_feat) in feature_sets.items():
        print("=" * 60)
        print("Feature Set:", feature_name)
        print("=" * 60)

        models = get_models()

        for model_name, model in models.items():
            print(f"\nModel: {model_name}")
            print("-" * 40)

            # Train
            model.fit(X_train_feat, y_train)

            # Predict & Evaluate
            y_pred = model.predict(X_test_feat)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

            print("Model Fit Check:")
            check_fit(model, X_train_feat, y_train, X_test_feat, y_test)

            # Save model to disk
            file_name = (
                feature_name.replace("-", "_").lower()
                + "_"
                + model_name.lower()
                + ".pkl"
            )
            save_path = f"{models_dir}{file_name}"
            joblib.dump(model, save_path)
            print(f"{file_name} saved successfully")

            trained_models[(feature_name, model_name)] = model

    return trained_models

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_sets = extract_and_save_features(X_train, X_test)
    train_and_evaluate(feature_sets, y_train, y_test)
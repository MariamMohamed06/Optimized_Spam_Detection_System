"""Preprocessing module for SMS spam data."""

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Lowercase text, replace punctuation/special characters with spaces, and normalize whitespace."""
    text = str(text).lower()
    # Keep alphanumeric characters, underscores, and whitespace; replace others with a space
    cleaned = "".join(
        char if (char.isalnum() or char == "_" or char.isspace()) else " "
        for char in text
    )
    # Split and rejoin to collapse consecutive whitespace and strip edges
    return " ".join(cleaned.split())


def load_and_preprocess_data(input_path: str = "../data/spam.csv",) -> pd.DataFrame:
    """Load raw dataset, drop unused columns, remove duplicates/empty texts, and clean messages."""
    df = pd.read_csv(input_path, encoding="latin-1")

    # Drop unnecessary columns if present
    cols_to_drop = [col for col in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Rename columns to standard names
    df = df.rename(columns={"v1": "label", "v2": "message"})

    # Deduplicate raw entries
    df = df.drop_duplicates()

    # Clean text
    df["clean_message"] = df["message"].apply(clean_text)

    # Filter out empty messages post-cleaning
    df = df[~df["clean_message"].str.strip().eq("")].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split preprocessed data into stratified train and test sets."""
    X = df["clean_message"]
    y = df["label"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main():
    input_file = "../data/spam.csv"
    output_file = "../data/cleaned_spam.csv"

    print("Loading data from " + str(input_file) + "...")
    df = load_and_preprocess_data(input_file)

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print("Cleaned dataset saved to " + str(output_file) + " (Shape: " + str(df.shape) + ")")

    # Perform split check
    X_train, X_test, y_train, y_test = split_data(df)
    print("Training samples: " + str(len(X_train)))
    print("Testing samples: " + str(len(X_test)))


if __name__ == "__main__":
    main()
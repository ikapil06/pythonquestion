"""
A complete end‑to‑end example for working with the Hotel Reservations
dataset.  This script reads the provided training and test CSV files,
cleans the data, performs a few exploratory data analysis steps (EDA),
encodes categorical features, trains a machine learning model and
generates predictions for the test set.  The resulting predictions are
written to a submission CSV file.  You can run this script as a
stand‑alone program or adapt pieces of it for your own notebooks.

Prerequisites
-------------
This script assumes the following files live in the same folder as the
script itself or in the current working directory when you execute
it:

* ``train.csv`` – the labelled training data.  Must include a
  ``booking_status`` column indicating whether the reservation was
  cancelled (1) or not (0).
* ``test.csv``  – the unlabelled test data.  Has the same feature
  columns as ``train.csv`` but no target column.

The code uses pandas for data manipulation, seaborn/matplotlib for
exploratory visualisations and scikit‑learn for model building.  If
you need to install any of these packages, use ``pip install
pandas seaborn matplotlib scikit‑learn``.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and test data from CSV files.

    Parameters
    ----------
    train_path : str
        Path to the training CSV file.
    test_path : str
        Path to the test CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the loaded training and test DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform simple data cleaning on the dataset.

    This function removes duplicate rows and fills missing values.  For
    numerical features missing values are filled with the median of the
    column; for categorical features missing values are filled with
    the most frequent category (mode).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with duplicates removed and
        missing values imputed.
    """
    cleaned = df.copy()
    # Drop duplicate rows to reduce potential bias
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    # Identify numeric and categorical columns
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned.select_dtypes(include=["object"]).columns
    # Fill missing numeric values with median
    for col in numeric_cols:
        if cleaned[col].isnull().any():
            median_val = cleaned[col].median()
            cleaned[col].fillna(median_val, inplace=True)
    # Fill missing categorical values with the mode
    for col in categorical_cols:
        if cleaned[col].isnull().any():
            mode_val = cleaned[col].mode().iloc[0]
            cleaned[col].fillna(mode_val, inplace=True)
    return cleaned


def perform_eda(df: pd.DataFrame, target_col: str) -> None:
    """Run a few basic EDA steps and visualisations.

    This function prints dataset summary information, the first few
    rows, target class distribution and generates example plots.  You
    can extend or modify this function to include further analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyse.
    target_col : str
        The name of the target column used for classification.
    """
    print("DataFrame shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nSummary statistics (numerical):\n", df.describe())
    print("\nClass distribution:")
    print(df[target_col].value_counts(normalize=True))
    # Plot class distribution
    plt.figure(figsize=(5, 3))
    sns.countplot(x=target_col, data=df)
    plt.title("Target class distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    # Correlation heatmap for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation heatmap (numeric features)")
        plt.tight_layout()
        plt.show()


def build_and_evaluate_model(
    train_df: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
) -> tuple[Pipeline, float, float, float]:
    """Train a classification model and evaluate it on a validation split.

    The function splits the cleaned training data into a training and
    validation set using an 80/20 split, performs one‑hot encoding for
    categorical features and trains a balanced Random Forest
    classifier.  Basic evaluation metrics are returned.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training data containing feature columns and the
        target column.
    target_col : str
        Name of the target column in ``train_df``.
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    tuple[Pipeline, float, float, float]
        A tuple containing the fitted pipeline and the evaluation
        metrics (accuracy, precision, recall).
    """
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    # Preprocessor: OneHot encode categorical features, leave numerical as is
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    # Build pipeline with RandomForest classifier.  Class weights help
    # mitigate imbalance in the target.
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    # Evaluate on validation set
    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    print("Validation accuracy:", acc)
    print("Validation precision:", prec)
    print("Validation recall:", rec)
    print("Validation F1 score:", f1)
    print("\nClassification report:\n", classification_report(y_val, val_preds))
    return clf, acc, prec, rec


def train_on_full_data(train_df: pd.DataFrame, target_col: str, clf: Pipeline) -> Pipeline:
    """Retrain the model on the full cleaned training dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training data containing feature columns and the
        target column.
    target_col : str
        Name of the target column in ``train_df``.
    clf : Pipeline
        The pipeline (with preprocessor and model) that should be
        refitted on the full data.

    Returns
    -------
    Pipeline
        The fitted pipeline trained on the entire dataset.
    """
    X_full = train_df.drop(columns=[target_col])
    y_full = train_df[target_col]
    clf.fit(X_full, y_full)
    return clf


def make_predictions(
    clf: Pipeline,
    test_df: pd.DataFrame,
    id_col: str | None = None,
    label_map: dict[int, str] | None = None,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Generate predictions on the test set and write them to a CSV file.

    If your test set contains a unique identifier column (such as
    ``Booking_ID``) then pass its name via ``id_col``.  The
    ``label_map`` argument allows you to translate numeric model
    predictions back to the original string labels if desired.  The
    resulting DataFrame is also returned.

    Parameters
    ----------
    clf : Pipeline
        Fitted pipeline used to generate predictions.
    test_df : pd.DataFrame
        The test DataFrame containing the same feature columns as the
        training data (minus the target).
    id_col : str | None, optional
        Name of the identifier column.  If provided, this column will
        be included in the submission.  Otherwise a simple index is
        used.  Defaults to None.
    label_map : dict[int, str] | None, optional
        Mapping from numeric class predictions to original labels.  If
        provided, predictions will be mapped accordingly; otherwise
        numeric predictions are used directly.  Defaults to None.
    output_path : str, optional
        Path where the submission CSV will be written, by default
        ``"submission.csv"``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the prediction results.
    """
    # Generate numeric predictions
    preds = clf.predict(test_df)
    # Translate numeric predictions to original labels if requested
    if label_map is not None:
        pred_labels = [label_map[int(p)] for p in preds]
    else:
        pred_labels = preds
    # Build the submission DataFrame
    if id_col and id_col in test_df.columns:
        submission_df = pd.DataFrame({
            id_col: test_df[id_col],
            "booking_status": pred_labels,
        })
    else:
        submission_df = pd.DataFrame({
            "booking_status": pred_labels,
        })
    # Write to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file written to {output_path}")
    return submission_df


def main() -> None:
    """Run the full workflow: load, clean, EDA, train, predict and save."""
    # Define paths to the training and test data.  Adjust as needed.
    train_path = os.path.join(os.getcwd(), "train.csv")
    test_path = os.path.join(os.getcwd(), "test.csv")
    # Load the data
    train_df, test_df = load_data(train_path, test_path)
    # Clean the data
    train_clean = clean_data(train_df)
    test_clean = clean_data(test_df)
    # Perform basic EDA
    perform_eda(train_clean, target_col="booking_status")
    # Train and evaluate model on a validation split
    clf, acc, prec, rec = build_and_evaluate_model(train_clean, target_col="booking_status")
    # Fit on full training data
    clf = train_on_full_data(train_clean, target_col="booking_status", clf=clf)
    # For this dataset the target is encoded as 1 for cancelled and 0
    # for not cancelled; we can map them back to strings if desired.
    label_map = {1: "Canceled", 0: "Not_Canceled"}
    # Generate predictions on the cleaned test data and save to CSV
    make_predictions(
        clf,
        test_clean,
        id_col=None,  # Set to 'Booking_ID' if your test set has this column
        label_map=label_map,
        output_path="submission.csv",
    )


if __name__ == "__main__":
    main()
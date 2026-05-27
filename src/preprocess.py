import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_iris(path="data/raw/iris.csv"):
    df = pd.read_csv(path)
    print("=== IRIS DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['species'].value_counts()}")
    return df


def preprocess_iris(path="data/raw/iris.csv"):
    df = load_iris(path)

    X = df.drop("species", axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df["species"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Classes: {list(le.classes_)}")
    return X_train, X_test, y_train, y_test, le.classes_


def load_titanic(path="data/raw/titanic.csv"):
    df = pd.read_csv(path)
    print("\n=== TITANIC DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nSurvival distribution:\n{df['Survived'].value_counts()}")
    return df


def preprocess_titanic(path="data/raw/titanic.csv"):
    df = load_titanic(path)

    # select useful features only
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[features + ["Survived"]].copy()

    # encode Sex column
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # encode Embarked column
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Features used: {list(X.columns)}")
    return X_train, X_test, y_train, y_test


def get_numeric_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


if __name__ == "__main__":
    preprocess_iris()
    preprocess_titanic()
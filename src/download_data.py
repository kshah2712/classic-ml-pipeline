import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

# Iris dataset
iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(iris_url)
iris_df.to_csv("data/raw/iris.csv", index=False)
print(f"Iris dataset downloaded → {iris_df.shape[0]} rows, {iris_df.shape[1]} columns")

# Titanic dataset
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(titanic_url)
titanic_df.to_csv("data/raw/titanic.csv", index=False)
print(f"Titanic dataset downloaded → {titanic_df.shape[0]} rows, {titanic_df.shape[1]} columns")

print("\nDone! Check data/raw/ folder")
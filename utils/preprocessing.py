import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

  
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor

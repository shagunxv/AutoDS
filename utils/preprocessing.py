import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values

    for col in X.columns:

        if pd.api.types.is_numeric_dtype(X[col]):

            X[col] = X[col].fillna(
                X[col].median()
            )

        else:

            X[col] = X[col].fillna(
                X[col].mode()[0]
            )

    # Encode categorical columns

    for col in X.select_dtypes(
        include=["object"]
    ).columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

    # Encode target if classification

    if y.dtype == "object":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y.astype(str)
        )

    return X, y
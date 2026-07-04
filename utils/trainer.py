import gc
from sklearn.linear_model import(
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from sklearn.neighbors import(
    KNeighborsClassifier,
    KNeighborsRegressor
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import joblib
import os
import time

def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        problem_type):

    results = {}

    if problem_type == "Classification":

        models = {
            "Logistic Regression":
                LogisticRegression(max_iter=1000,random_state=42),

            "Decision Tree":
                DecisionTreeClassifier(random_state=42,max_depth=10),

            "Random Forest":
                RandomForestClassifier(
    n_estimators=50,
    random_state=42
),
            "K-Nearest Neighbors" :
                KNeighborsClassifier(n_neighbors=5),
            "Gradient Boosting":
                GradientBoostingClassifier(random_state=42,n_estimators=20),

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            del predictions
            gc.collect()

        best_model = max(
            results,
            key=lambda x:
            results[x]["accuracy"]
        )

        best_score = results[
            best_model
        ]["accuracy"]

    else:

        models = {
            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(random_state=42,max_depth=10),

            "Random Forest":
                RandomForestRegressor(random_state=42,max_depth=10,n_estimators=20,n_jobs=1),
            "Ridge Regression":
                Ridge(),
            "Lasso Regression":
                Lasso(),
            "KNN Regressor":
                KNeighborsRegressor(n_neighbors=5),
            "Gradient Boosting":
                GradientBoostingRegressor(random_state=42,n_estimators=50),
            "Extra Trees":
                ExtraTreesRegressor(random_state=42,n_estimators=20,max_depth=10,n_jobs=1)
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            r2 = r2_score(
                y_test,
                predictions
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = (
                mean_squared_error(
                    y_test,
                    predictions
                ) ** 0.5
            )

            results[name] = {
                "r2": r2,
                "mae": mae,
                "rmse": rmse
            }
            del predictions
            gc.collect()

        best_model = max(
            results,
            key=lambda x:
            results[x]["r2"]
        )

        best_score = results[
            best_model
        ]["r2"]

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    joblib.dump(
        models[best_model],
        "saved_models/best_model.pkl"
    )
    import matplotlib.pyplot as plt

# Save Best Model

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    joblib.dump(
        models[best_model],
        "saved_models/best_model.pkl"
    )

# Feature Importance Graph

    best_model_object = models[
        best_model
    ]

    if hasattr(
        best_model_object,
        "feature_importances_"
    ):

        os.makedirs(
            "static/plots",
            exist_ok=True
        )

        importance = (
            best_model_object
            .feature_importances_
        )

        feature_names = (
            X_train.columns
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            feature_names,
            importance
        )

        plt.xlabel(
            "Importance"
        )

        plt.ylabel(
            "Features"
        )

        plt.title(
            "Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            "static/plots/feature_importance.png"
        )

        plt.close()

    return (
        results,
        best_model,
        best_score
    )
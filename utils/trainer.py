import os
import joblib
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from utils.visualizer import generate_actual_vs_predicted

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RandomizedSearchCV
)
from sklearn.base import clone
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
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
from sklearn.neighbors import (
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

from utils.preprocessing import build_preprocessor


def generate_feature_importance(best_pipeline):

    import matplotlib.pyplot as plt
    import numpy as np

    plot_path = "static/plots/feature_importance.png"

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance = np.mean(
            np.abs(model.coef_),
            axis=0
        )
        if importance.ndim > 1:
            importance = importance.mean(axis=0)

    else:
        if os.path.exists(plot_path):
            os.remove(plot_path)
        return

    feature_names = preprocessor.get_feature_names_out()

    if len(feature_names) != len(importance):
        if os.path.exists(plot_path):
            os.remove(plot_path)
        return

    indices = np.argsort(importance)[-15:]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        feature_names[indices],
        importance[indices],
        color="#4285F4"
    )

    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)



def generate_confusion_matrix(y_test, predictions):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    fig.tight_layout()
    fig.savefig(
        "static/plots/confusion_matrix.png",
        dpi=150
    )
    plt.close(fig)

def generate_classification_report(y_test, predictions):

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    return report




def generate_roc_curve(best_pipeline, X_test, y_test):

    import matplotlib.pyplot as plt

    model = best_pipeline.named_steps["model"]

    classes = model.classes_

    if len(classes) != 2:
        return

    if hasattr(best_pipeline, "predict_proba"):
        scores = best_pipeline.predict_proba(X_test)[:, 1]

    elif hasattr(best_pipeline, "decision_function"):
        scores = best_pipeline.decision_function(X_test)

    else:
        return

    fpr, tpr, thresholds = roc_curve(
        y_test,
        scores,
        pos_label=classes[1]
    )

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr, tpr,
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
        linewidth=2
    )

    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig("static/plots/roc_curve.png", dpi=150)
    plt.close(fig)





def get_cv_strategy(y_train, problem_type):
    if problem_type == "Classification":
        min_class_count = y_train.value_counts().min()
        n_splits = min(5, min_class_count)

        if n_splits < 2:
            raise ValueError(
                "Not enough samples in each class for cross-validation."
            )

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

def get_parameter_grid(problem_type):

    if problem_type == "Classification":
        return {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }

    return {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }



def train_models(X_train, X_test, y_train, y_test, problem_type):

    results = {}
    trained_models = {}
    classification_report_data = None

    if problem_type == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42, max_depth=10
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=50, random_state=42
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42, n_estimators=20
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=50, random_state=42
            ),
            "Support Vector Machine": SVC(
                random_state=42
            )
        }

    else:

        models = {
            "Linear Regression": LinearRegression(),

            "Decision Tree": DecisionTreeRegressor(
                random_state=42, max_depth=10
            ),

            "Random Forest": RandomForestRegressor(
                random_state=42,
                max_depth=10,
                n_estimators=50,
                n_jobs=1
            ),

            "Ridge Regression": Ridge(),

            "Lasso Regression": Lasso(),

            "KNN Regressor": KNeighborsRegressor(
                n_neighbors=5
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42, n_estimators=50
            ),

            "Extra Trees": ExtraTreesRegressor(
                random_state=42,
                n_estimators=50,
                max_depth=10,
                n_jobs=1
            )
        }

    cv = get_cv_strategy(y_train, problem_type)

    scoring = (
        "f1_weighted"
        if problem_type == "Classification"
        else "r2"
    )
    
    for name, model in models.items():

        try:

            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model)
            ])

            if name == "Random Forest":
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=get_parameter_grid(problem_type),
                    n_iter=8,
                    scoring=scoring,
                    cv=cv,
                    random_state=42,
                    n_jobs=1
                )

                search.fit(X_train, y_train)

                pipeline = search.best_estimator_
                cv_mean = search.best_score_

            else:
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1
                )

                cv_mean = cv_scores.mean()
                pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            # if problem_type == "Classification":
            #     generate_confusion_matrix(y_test, predictions)

            if problem_type == "Classification":
                results[name] = {
                    "cv_score": cv_mean,
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "f1": f1_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0
                    )
                }

            else:
                results[name] = {
                    "cv_score": cv_mean,
                    "r2": r2_score(y_test, predictions),
                    "mae": mean_absolute_error(y_test, predictions),
                    "rmse": mean_squared_error(y_test, predictions) ** 0.5
                }

            trained_models[name] = pipeline

        except Exception as e:

            print(f"Error training {name}: {e}")

    if not trained_models:
        raise ValueError(
            "All models failed during training. "
            "Please check your dataset."
        )

    if problem_type == "Classification":

        best_model = max(
            results,
            key=lambda name: results[name]["cv_score"]
        )

        best_score = results[best_model]["cv_score"]

    else:

        best_model = max(
            results,
            key=lambda name: results[name]["r2"]
        )

        best_score = results[best_model]["r2"]

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(
        trained_models[best_model],
        "saved_models/best_model.pkl"
    )


    if problem_type == "Classification":
        best_predictions = trained_models[best_model].predict(X_test)


        classification_report_data = generate_classification_report(
            y_test,
            best_predictions
        )

        generate_confusion_matrix(
            y_test,
            best_predictions
        )

        generate_roc_curve(
            trained_models[best_model],
            X_test,
            y_test
        )



    generate_feature_importance(
        trained_models[best_model]
    )

    if problem_type == "Regression":

        best_predictions = trained_models[best_model].predict(X_test)

        generate_actual_vs_predicted(
            y_test,
            best_predictions
        )


    return results, best_model, best_score, classification_report_data

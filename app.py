
from flask import Flask, render_template, request, send_file
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from utils.trainer import train_models

from utils.visualizer import (
    generate_heatmap,
    generate_missing_plot,
    generate_target_distribution,
    generate_actual_vs_predicted
)


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)


# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")


# UPLOAD DATASET
@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files.get("dataset")

    if not file or not file.filename:
        return "Please upload a CSV file.", 400

    if not file.filename.lower().endswith(".csv"):
        return "Only CSV files are supported.", 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

    except Exception:
        return "Unable to read the uploaded CSV file.", 400

    if df.empty or len(df.columns) < 2:
        return "Dataset must contain at least two columns and some rows.", 400

    # Generate dataset visualizations
    generate_heatmap(df)
    generate_missing_plot(df)

    # Dataset information
    rows, cols = df.shape

    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    numerical_cols = len(
        df.select_dtypes(include=["number"]).columns
    )

    categorical_cols = len(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    preview_table = df.head().to_html(
        classes="table",
        index=False
    )

    return render_template(
        "preview.html",
        rows=rows,
        cols=cols,
        missing_values=missing_values,
        duplicates=duplicates,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        preview_table=preview_table,
        column_names=df.columns.tolist(),
        filepath=filepath
    )


# DATASET ANALYSIS
@app.route("/analyze", methods=["POST"])
def analyze():

    target = request.form.get("target")
    selected_features = request.form.getlist("features")
    filepath = request.form.get("filepath")

    if not target or not filepath or not selected_features:
        return "Please select a target and at least one feature.", 400

    if not os.path.isfile(filepath):
        return "Dataset file not found.", 404

    df = pd.read_csv(filepath)

    if target not in df.columns:
        return "Invalid target column.", 400

    if target in selected_features:
        return "Target cannot also be a feature.", 400

    if not set(selected_features).issubset(df.columns):
        return "Invalid feature selection.", 400

    # Select required columns
    df = df[selected_features + [target]].copy()

    # Remove missing target rows
    df = df.dropna(subset=[target])

    if df.empty:
        return "No valid rows remain after removing missing targets.", 400

    # Detect problem type
    y = df[target]

    if (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
        or y.nunique() <= 20
    ):
        problem_type = "Classification"

    else:
        problem_type = "Regression"

    # Limit dataset size
    if len(df) > 1500:
        df = df.sample(n=1500, random_state=42)

    X = df[selected_features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train and evaluate models
    results, best_model, best_score, classification_report_data = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        problem_type
    )

    # Target distribution visualization
    generate_target_distribution(df, target)

    # Dashboard
    return render_template(
        "analysis.html",
        target=target,
        selected_features=selected_features,
        problem_type=problem_type,
        total_rows=df.shape[0],
        total_columns=df.shape[1],
        train_rows=X_train.shape[0],
        test_rows=X_test.shape[0],
        feature_count=X.shape[1],
        results=results,
        best_model=best_model,
        best_score=best_score,
        classification_report_data=classification_report_data
    )


# DOWNLOAD TRAINED MODEL
@app.route("/download-model")
def download_model():

    model_path = "saved_models/best_model.pkl"

    if not os.path.exists(model_path):
        return "No trained model available.", 404

    return send_file(
        model_path,
        as_attachment=True
    )


# RUN APPLICATION
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

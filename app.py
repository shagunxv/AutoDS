from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess_data
from utils.trainer import train_models
import pandas as pd
import os
from flask import send_file
from utils.visualizer import (
    generate_heatmap,
    generate_missing_plot,
    generate_target_distribution
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files["dataset"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    df = pd.read_csv(filepath)
    generate_heatmap(df)

    generate_missing_plot(df)

   

    rows = df.shape[0]
    cols = df.shape[1]

    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    numerical_cols = len(
        df.select_dtypes(include=["number"]).columns
    )

    categorical_cols = len(
        df.select_dtypes(include=["object"]).columns
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
@app.route("/analyze", methods=["POST"])
def analyze():

    target = request.form["target"]

    selected_features = request.form.getlist(
    "features"
    )

    filepath = request.form["filepath"]

    df = pd.read_csv(filepath)
    generate_target_distribution(
        df,
        target
    )

    y = df[target]

    if y.dtype == "object":
        problem_type = "Classification"

    elif y.nunique() <= 20:
        problem_type = "Classification"

    else:
        problem_type = "Regression"

    X= df[selected_features]
    y=df[target]
    temp_df=pd.concat(
        [X,y],
        axis=1
    )
    X,y=preprocess_data(temp_df,target)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results, best_model, best_score = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        problem_type
    )

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
    best_score=best_score
)
@app.route("/download-model")
def download_model():
    return send_file(
        "saved_models/best_model.pkl",
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
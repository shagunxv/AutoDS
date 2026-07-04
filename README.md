# 🚀 AutoDS

### Automated Machine Learning Platform

AutoDS is a Flask-based Automated Machine Learning platform that allows users to upload a dataset, select target and feature columns, automatically preprocess data, detect the machine learning problem type, compare multiple models, visualize the dataset, and download the best trained model.

---

## ✨ Features

- 📂 Upload CSV datasets
- 🎯 Select Target Variable
- 📊 Select Independent Features
- 🤖 Automatic Classification/Regression Detection
- 🧹 Automatic Data Preprocessing
- 📈 Model Comparison
- 🏆 Automatic Best Model Selection
- 💾 Download Trained Model (.pkl)
- 📉 Correlation Heatmap
- 📊 Missing Values Visualization
- 📈 Target Distribution Plot
- 🎨 Modern Dark UI inspired by Apple/OpenAI

---

## 🛠 Tech Stack

### Frontend

- HTML5
- CSS3

### Backend

- Flask
- Python

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

---

## 🤖 Supported Machine Learning Models

### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting
- Extra Trees Classifier

### Regression

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

---

## 📷 Screenshots

### 🏠 Landing Page

> "C:\Users\shagu\OneDrive\Pictures\Screenshots\Screenshot (74).png"

---

### 📂 Dataset Preview

> "C:\Users\shagu\OneDrive\Pictures\Screenshots\Screenshot (75).png"
> "C:\Users\shagu\OneDrive\Pictures\Screenshots\Screenshot (76).png"

---

### 📊 Model Comparison
 
> "C:\Users\shagu\OneDrive\Pictures\Screenshots\Screenshot (77).png"

---

### 📈 Analysis Dashboard

> "C:\Users\shagu\OneDrive\Pictures\Screenshots\Screenshot (78).png" 

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AutoDS.git
```

Move into the project folder

```bash
cd AutoDS
```

Create a virtual environment

```bash
python -m venv venv
```

Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```
AutoDS/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── static/
│   ├── css/
│   └── plots/
│
├── templates/
│   ├── index.html
│   ├── preview.html
│   └── analysis.html
│
├── uploads/
│
├── saved_models/
│
└── utils/
    ├── preprocessing.py
    ├── trainer.py
    └── visualizer.py
```

---

## 🚀 Future Improvements

- PDF Analysis Report
- Model Leaderboard
- Feature Importance Chart
- ROC Curve
- Confusion Matrix
- Actual vs Predicted Plot
- Hyperparameter Tuning
- Deployment on Render
- User Authentication
- Drag & Drop Dataset Upload

---

## 👩‍💻 Author

**Shagun Ojha**

BCA (Machine Learning & Data Science)

GitHub:
https://github.com/shagunxv

---

## ⭐ If you found this project useful...

Please consider giving it a ⭐ on GitHub!
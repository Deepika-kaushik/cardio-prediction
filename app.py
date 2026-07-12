from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
 
import seaborn as sns

app = Flask(__name__)

import os

if not os.path.exists("cardio_model.pkl"):
    import train_model

df_model = pd.read_csv("cardio_train.csv", sep=";")

X = df_model.drop(["id", "cardio"], axis=1)
y = df_model["cardio"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/dashboard")
def dashboard():
    df = pd.read_csv("cardio_train.csv", sep=";")

    total_records = len(df)
    total_features = len(df.columns)
    missing_values = df.isnull().sum().sum()

    table = df.head(10).to_html(classes='table', index=False)
    cardio_counts =df['cardio'].value_counts()

    plt.figure(figsize=(5,5))
    cardio_counts.plot(kind='bar')
    plt.title('Cardiovascular Disease Distribution')
    plt.xlabel('0 = No Disease, 1 = Disease')
    plt.ylabel('Number of Patients')

    plt.savefig('static/cardio_distribution.png')
    plt.close()
    # Age Distribution Chart
    plt.figure(figsize=(8,5))
    df['age'].hist(bins=30)

    plt.title('Age Distribution')
    plt.xlabel('Age (Days)')
    plt.ylabel('Number of Patients')

    plt.savefig('static/age_distribution.png')
    plt.close()
    # Gender Distribution Chart
    plt.figure(figsize=(6,4))

    df['gender'].value_counts().plot(kind='bar')

    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')

    plt.savefig('static/gender_distribution.png')
    plt.close()
    # Blood Pressure Distribution Chart
    plt.figure(figsize=(8,5))
    df['ap_hi'].hist(bins=30)

    plt.title('Systolic Blood Pressure Distribution')
    plt.xlabel('Blood Pressure')
    plt.ylabel('Number of Patients')

    plt.savefig('static/bp_distribution.png')
    plt.close()
    # Cholesterol Distribution Chart
    plt.figure(figsize=(6,4))

    df['cholesterol'].value_counts().sort_index().plot(kind='bar')

    plt.title('Cholesterol Distribution')
    plt.xlabel('Cholesterol Level')
    plt.ylabel('Count')

    plt.savefig('static/cholesterol_distribution.png')
    plt.close()

    # Correlation Matrix

    plt.figure(figsize=(12,8))

    corr_matrix = df.corr(numeric_only=True)

    sns.heatmap(
    corr_matrix,
    annot=False,
    cmap='coolwarm'
)

    plt.title('Correlation Matrix')

    plt.savefig('static/correlation_matrix.png')

    plt.close()

    return render_template(
        "dashboard.html",
        total_records=total_records,
        total_features=total_features,
        missing_values=missing_values,
        table=table,
        chart="cardio_distribution.png",
        age_chart="age_distribution.png",
        gender_chart="gender_distribution.png",
        bp_chart="bp_distribution.png",
        chol_chart="cholesterol_distribution.png",
        corr_chart="correlation_matrix.png"
    )
@app.route("/analysis")
def analysis():
    return render_template("analysis.html")
@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None

    if request.method == "POST":

        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = int(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        data = np.array([[
            age,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active
        ]])

        result = model.predict(data)[0]

        if result == 1:
            prediction = "High Risk of Cardiovascular Disease"
        else:
            prediction = "Low Risk of Cardiovascular Disease"

    return render_template(
        "predict.html",
        prediction=prediction
    )
if __name__ == "__main__":
    app.run(debug=True)

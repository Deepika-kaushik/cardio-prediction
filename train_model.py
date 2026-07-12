import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import joblib

# Load dataset
df = pd.read_csv("cardio_train.csv", sep=";")

# Features and target
X = df.drop(["id", "cardio"], axis=1)
y = df["cardio"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# Print Results
print("\nModel Accuracies\n")

print("Logistic Regression :", round(lr_acc * 100, 2), "%")
print("KNN                 :", round(knn_acc * 100, 2), "%")
print("Decision Tree       :", round(dt_acc * 100, 2), "%")
print("Random Forest       :", round(rf_acc * 100, 2), "%")
print("SVM                 :", round(svm_acc * 100, 2), "%")

# Save best model (Random Forest)
joblib.dump(rf, "cardio_model.pkl")

print("\nBest model saved as cardio_model.pkl")
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit UI
st.title("💓 Heart Disease Detection App")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

st.subheader("Enter Patient Details:")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-anginal Pain", 4:"Asymptomatic"}[x])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
restecg = st.selectbox("Resting ECG Results", [0, 1, 2], format_func=lambda x: {0:"Normal", 1:"ST-T Wave Abnormality", 2:"Left Ventricular Hypertrophy"}[x])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [1, 2, 3], format_func=lambda x: {1:"Upward", 2:"Flat", 3:"Downward"}[x])

# Prediction Button
if st.button("Predict"):
    user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]]
    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)[0]
    result = "Heart Disease" if prediction == 1 else "Normal"
    st.success(f"Prediction: {result}")
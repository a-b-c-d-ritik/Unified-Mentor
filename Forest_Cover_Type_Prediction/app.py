import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("forest_model.pkl")
DATA_PATH = Path("train.csv")  # change if running locally

COVER_TYPE_LABELS = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

st.set_page_config(page_title="Forest Cover Type Prediction", layout="centered")
st.title("🌲 Forest Cover Type Prediction")

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def train_model(df):
    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)

    joblib.dump({"model": clf, "columns": X.columns.tolist()}, MODEL_PATH)

    return acc, report

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

# Sidebar
mode = st.sidebar.radio("Mode", ["Train Model", "Predict"])

df = load_data()

if mode == "Train Model":
    st.subheader("Train a new model")
    if st.button("Start Training"):
        acc, report = train_model(df)
        st.success(f"Training completed — Accuracy: {acc:.4f}")
        st.text("Classification Report:")
        st.text(report)

else:
    st.subheader("Enter Feature Values")
    model_bundle = load_model()
    if model_bundle is None:
        st.warning("Model not found. Please train it first.")
        st.stop()

    feature_values = {}
    for col in model_bundle["columns"]:
        if df[col].nunique() <= 2:
            feature_values[col] = st.selectbox(col, [0, 1])
        else:
            min_val, max_val = int(df[col].min()), int(df[col].max())
            mean_val = int(df[col].mean())
            feature_values[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)

    if st.button("Predict Cover Type"):
        X_input = pd.DataFrame([feature_values])
        pred_class = model_bundle["model"].predict(X_input)[0]
        st.success(f"Predicted Cover Type: {pred_class} — {COVER_TYPE_LABELS[pred_class]}")

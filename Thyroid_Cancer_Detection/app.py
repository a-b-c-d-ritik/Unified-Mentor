import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

MODEL_PATH = Path("model_pipeline.pkl")
DATA_PATH = Path("dataset.csv")  # your uploaded CSV

st.set_page_config(page_title="Thyroid Cancer Recurrence Predictor", layout="centered")

st.title("Thyroid Cancer Recurrence — Demo")
st.markdown(
    "Upload / use the provided dataset and train a model, or load an existing model. "
    "The app predicts whether a past thyroid-cancer patient is likely to experience recurrence."
)

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Basic cleaning: strip column whitespace
    df.columns = [c.strip() for c in df.columns]
    return df

def build_pipeline(cat_cols, num_cols):
    # Impute and encode pipeline
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])
    return pipeline

def train_and_save(df):
    # Target column assumed 'Recurred' (if named differently, adjust)
    target_col = "Recurred"
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
        st.stop()
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).map(lambda v: 1 if str(v).strip().lower() in ["yes","1","true","recurred","y"] else 0)

    # Decide categorical vs numeric columns heuristically
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipeline = build_pipeline(cat_cols, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with st.spinner("Training model..."):
        pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save
    joblib.dump({
        "pipeline": pipeline,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target_col": target_col
    }, MODEL_PATH)

    return {"acc": acc, "report": report, "cm": cm, "pipeline": pipeline, "cat_cols": cat_cols, "num_cols": num_cols}

def load_model():
    if MODEL_PATH.exists():
        data = joblib.load(MODEL_PATH)
        return data
    return None

# --- UI ---
st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ["Predict (use model)", "Train model", "Quick EDA"])

if not DATA_PATH.exists():
    st.error(f"Dataset file not found at {DATA_PATH}. Please upload or place the CSV there.")
    st.stop()

df = load_data()

if mode == "Quick EDA":
    st.subheader("Dataset preview")
    st.dataframe(df.head(50))
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values per column:")
    st.write(df.isna().sum())
    st.markdown("**Value counts for categorical columns**")
    for c in df.select_dtypes(exclude=[np.number]).columns:
        st.write(f"--- **{c}**")
        st.write(df[c].value_counts(dropna=False).head(20))
    st.stop()

if mode == "Train model":
    st.subheader("Train model on uploaded dataset")
    st.write("Dataset shape:", df.shape)
    if st.button("Start training"):
        res = train_and_save(df)
        st.success(f"Training complete — test accuracy: {res['acc']:.4f}")
        st.subheader("Classification report (test)")
        st.write(pd.DataFrame(res["report"]).transpose())
        st.subheader("Confusion Matrix")
        cm = res["cm"]
        fig, ax = plt.subplots()
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center", color="white")
        st.pyplot(fig)
        st.info(f"Model pipeline saved to {MODEL_PATH}")
    st.stop()

# mode == "Predict (use model)"
st.subheader("Predict recurrence for a single patient")

model_bundle = load_model()
if model_bundle is None:
    st.warning("No trained model found. Please train a model first (switch to 'Train model' in the sidebar).")
    if st.button("Train now (quick)"):
        res = train_and_save(df)
        model_bundle = load_model()
        st.success("Model trained and saved. Switching to predict.")
    else:
        st.stop()

pipeline = model_bundle["pipeline"]
cat_cols = model_bundle["cat_cols"]
num_cols = model_bundle["num_cols"]
target_col = model_bundle["target_col"]

st.write("Enter patient data (leave empty / default for unknown):")

# Build inputs dynamically using dataset unique values where possible
input_data = {}
for col in num_cols:
    # numeric input
    col_min = float(df[col].min()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else 0.0
    col_max = float(df[col].max()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else 100.0
    default = float(df[col].median()) if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) else 0.0
    input_data[col] = st.number_input(col, min_value=col_min, max_value=col_max, value=default)

for col in cat_cols:
    options = df[col].dropna().unique().astype(str).tolist() if col in df.columns else ["No", "Yes"]
    # ensure "Unknown" option exists
    if "Unknown" not in options:
        options = ["Unknown"] + options
    input_data[col] = st.selectbox(col, options)

if st.button("Predict"):
    X_input = pd.DataFrame([input_data])
    # align columns order
    # ensure numeric columns types match
    for nc in num_cols:
        X_input[nc] = pd.to_numeric(X_input[nc], errors="coerce")
    pred = pipeline.predict(X_input)[0]
    pred_proba = pipeline.predict_proba(X_input)[0] if hasattr(pipeline, "predict_proba") else None
    label = "Recurred" if int(pred) == 1 else "No recurrence"
    st.markdown(f"## Prediction: **{label}**")
    if pred_proba is not None:
        st.write(f"Probability (No recurrence / Recurred): {pred_proba}")
    # show feature importances if RandomForest
    if hasattr(pipeline.named_steps["clf"], "feature_importances_"):
        # Get feature names after preprocessing
        preproc = pipeline.named_steps["preproc"]
        # numeric feature names
        feature_names = []
        if hasattr(preproc, "transformers_"):
            for name, trans, cols in preproc.transformers_:
                if name == "num":
                    feature_names.extend(cols)
                elif name == "cat":
                    # get categories from onehot
                    ohe = trans.named_steps["onehot"]
                    cat_cols_local = cols
                    try:
                        cat_names = ohe.get_feature_names_out(cat_cols_local).tolist()
                    except:
                        # sklearn older versions
                        cat_names = []
                    feature_names.extend(cat_names)
        importances = pipeline.named_steps["clf"].feature_importances_
        # take top 10
        top_idx = np.argsort(importances)[::-1][:10]
        fig2, ax2 = plt.subplots()
        ax2.barh(np.array(feature_names)[top_idx][::-1], importances[top_idx][::-1])
        ax2.set_title("Top feature importances")
        st.pyplot(fig2)

st.markdown("---")
st.caption("Model & preprocessing pipeline trained on your uploaded dataset. See project PDF for dataset description.")

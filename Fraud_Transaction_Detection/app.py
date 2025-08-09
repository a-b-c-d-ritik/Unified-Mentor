import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import datetime

# Paths
DATA_DIR = Path("./dataset")         # <--- place all your .pkl files here
MODEL_PATH = Path("fraud_model.pkl")

st.set_page_config(page_title="Fraud Transaction Detection", layout="centered")
st.title("💳 Fraud Transaction Detection (with automatic historical lookups)")

@st.cache_data(show_spinner=False)
def load_all_pickles(data_dir=DATA_DIR):
    """Load all .pkl files from the dataset directory and concat into a DataFrame."""
    files = sorted(glob.glob(str(data_dir / "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir.resolve()}.")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to read {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    # Ensure types and columns exist
    df.columns = [c.strip() for c in df.columns]
    # Expecting TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, TX_FRAUD
    if "TX_DATETIME" not in df.columns:
        # try fallback names
        raise KeyError("TX_DATETIME column not found in dataset files.")
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    # Ensure numeric
    df["TX_AMOUNT"] = pd.to_numeric(df["TX_AMOUNT"], errors="coerce").fillna(0.0)
    df["TX_FRAUD"] = pd.to_numeric(df["TX_FRAUD"], errors="coerce").fillna(0).astype(int)
    # sort so rolling time windows work
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)
    return df

def compute_rolling_features_for_training(df):
    """
    Compute time-windowed rolling features per customer and terminal to use for training.
    We'll compute the 7-day rolling fraud rate and avg amount per customer, and a 7-day
    fraud rate per terminal. We shift these by 1 transaction so current row doesn't see itself.
    """
    df = df.copy()
    # basic time features
    df["TX_DOW"] = df["TX_DATETIME"].dt.dayofweek
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour

    # For rolling by time-window, set index to TX_DATETIME temporarily per group
    # We'll compute per-customer rolling stats
    def cust_stats(g):
        g = g.sort_values("TX_DATETIME")
        g = g.set_index("TX_DATETIME")
        # rolling 7 days: fraud mean and amount mean; shift(1) to exclude current tx
        g["CUST_FRAUD_7D"] = g["TX_FRAUD"].rolling("7d").mean().shift(1).fillna(0.0)
        g["CUST_AMT_AVG_7D"] = g["TX_AMOUNT"].rolling("7d").mean().shift(1).fillna(0.0)
        return g.reset_index()

    def term_stats(g):
        g = g.sort_values("TX_DATETIME")
        g = g.set_index("TX_DATETIME")
        g["TERM_FRAUD_7D"] = g["TX_FRAUD"].rolling("7d").mean().shift(1).fillna(0.0)
        return g.reset_index()

    # apply groupby
    df = df.groupby("CUSTOMER_ID", group_keys=False).apply(cust_stats)
    df = df.groupby("TERMINAL_ID", group_keys=False).apply(term_stats)

    # If multiple groupby passes impacted order, ensure sort again
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)
    return df

def train_and_save_model(df):
    """Compute features, train a RandomForest, and save pipeline (model + feature list)."""
    df_fe = compute_rolling_features_for_training(df)
    # feature columns to use
    features = ["TX_AMOUNT", "TX_DOW", "TX_HOUR", "CUST_FRAUD_7D", "CUST_AMT_AVG_7D", "TERM_FRAUD_7D"]
    # Some rows (early rows) might still have NaNs after grouping; fill with zeros
    X = df_fe[features].fillna(0.0)
    y = df_fe["TX_FRAUD"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    report = classification_report(y_test, y_pred, zero_division=0)

    # Save model and features + training meta
    joblib.dump({
        "model": clf,
        "features": features,
        "trained_on_rows": len(df_fe)
    }, MODEL_PATH)

    return {"accuracy": acc, "auc": auc, "report": report}

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def lookup_historical_features(df, tx_datetime, customer_id, terminal_id, window_days=7):
    """
    For a single new transaction at tx_datetime, compute historical features
    using past transactions in df that occurred before tx_datetime.
    Returns a dict with CUST_FRAUD_7D, CUST_AMT_AVG_7D, TERM_FRAUD_7D.
    """
    # ensure timestamp
    if isinstance(tx_datetime, str):
        tx_datetime = pd.to_datetime(tx_datetime)
    start_window = tx_datetime - pd.Timedelta(days=window_days)

    # slice past transactions (strictly before tx_datetime)
    past = df[(df["TX_DATETIME"] < tx_datetime) & (df["TX_DATETIME"] >= start_window)]

    # customer stats
    cust_past = past[past["CUSTOMER_ID"] == customer_id]
    if len(cust_past):
        cust_fraud_rate = cust_past["TX_FRAUD"].mean()
        cust_amt_avg = cust_past["TX_AMOUNT"].mean()
    else:
        cust_fraud_rate = 0.0
        cust_amt_avg = 0.0

    # terminal stats
    term_past = past[past["TERMINAL_ID"] == terminal_id]
    if len(term_past):
        term_fraud_rate = term_past["TX_FRAUD"].mean()
    else:
        term_fraud_rate = 0.0

    return {
        "CUST_FRAUD_7D": float(cust_fraud_rate),
        "CUST_AMT_AVG_7D": float(cust_amt_avg),
        "TERM_FRAUD_7D": float(term_fraud_rate)
    }

# ------- UI -------
st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ["Train model (recompute features)", "Predict transaction (live lookup)"])

try:
    data_load_state = st.sidebar.text("Loading dataset from ./dataset ...")
    df_all = load_all_pickles(DATA_DIR)
    data_load_state.text(f"Loaded {len(df_all):,} rows from {DATA_DIR.resolve()}")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if mode == "Train model (recompute features)":
    st.header("Train model (this will recompute rolling features from historical data)")
    st.markdown("""
    The app will compute 7-day historical features per customer and per terminal (shifted so
    the current transaction doesn't leak into its own features), then train a RandomForest.
    """)
    if st.button("Start training"):
        with st.spinner("Computing features and training model..."):
            res = train_and_save_model(df_all)
        st.success(f"Training finished — accuracy: {res['accuracy']:.4f}" + (f", AUC: {res['auc']:.4f}" if res['auc'] else ""))
        st.text("Classification report (test):")
        st.text(res["report"])
        st.info(f"Model saved to {MODEL_PATH.resolve()}")
    st.markdown("---")
    st.write("Preview of data (first 200 rows):")
    st.dataframe(df_all.head(200))

else:
    st.header("Predict single transaction (automatic historical lookup)")

    model_bundle = load_model()
    if model_bundle is None:
        st.warning("No trained model found. Please go to 'Train model' and train a model first.")
        st.stop()

    # Prediction form
    st.subheader("Enter transaction details")
    col1, col2 = st.columns(2)
    with col1:
        tx_date = st.date_input("Transaction date", value=datetime.date(2018, 9, 30),
                                min_value=df_all["TX_DATETIME"].dt.date.min(),
                                max_value=df_all["TX_DATETIME"].dt.date.max())
        tx_time = st.time_input("Transaction time (hour:min)", value=datetime.time(12, 0))
        customer_id = st.text_input("Customer ID", value="")
        terminal_id = st.text_input("Terminal ID", value="")
    with col2:
        tx_amount = st.number_input("Transaction amount", min_value=0.0, value=50.0, step=0.01)
        window_days = st.slider("Historical window (days) used for lookup", min_value=1, max_value=30, value=7)

    # Combine date+time into Timestamp
    tx_dt = pd.to_datetime(datetime.datetime.combine(tx_date, tx_time))

    if st.button("Compute lookup & predict"):
        if customer_id == "" or terminal_id == "":
            st.error("Please provide both Customer ID and Terminal ID (they must match your dataset values).")
        else:
            # compute historical features automatically from df_all
            hist = lookup_historical_features(df_all, tx_dt, customer_id, terminal_id, window_days=window_days)
            st.write("Historical features (computed from dataset):")
            st.json(hist)

            # create model features vector
            feat = {
                "TX_AMOUNT": float(tx_amount),
                "TX_DOW": int(tx_dt.dayofweek),
                "TX_HOUR": int(tx_dt.hour),
                "CUST_FRAUD_7D": hist["CUST_FRAUD_7D"],
                "CUST_AMT_AVG_7D": hist["CUST_AMT_AVG_7D"],
                "TERM_FRAUD_7D": hist["TERM_FRAUD_7D"]
            }
            st.write("Model input features:")
            st.json(feat)

            # predict
            model = model_bundle["model"]
            features_order = model_bundle["features"]
            X_input = pd.DataFrame([feat])[features_order].fillna(0.0)
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

            st.markdown("### Result")
            st.write(f"Prediction: **{'FRAUD' if pred==1 else 'LEGITIMATE'}**")
            if proba is not None:
                st.write(f"Fraud probability: **{proba:.2%}**")
            # optional: show top feature importances
            if hasattr(model, "feature_importances_"):
                importances = pd.Series(model.feature_importances_, index=features_order).sort_values(ascending=False)
                st.subheader("Feature importances")
                st.bar_chart(importances)

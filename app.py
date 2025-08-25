import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
import os
import requests
import tempfile
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from urllib.parse import urlparse

# -------------------
# Configuration https://github.com/Nhlabatsi/phishing_detector/releases/download/v1.0.0/iso_forest.pkl
# -------------------
MODEL_URLS = {
    "rnn_model.pkl": "https://github.com/Nhlabatsi/phishing_detector/releases/download/v1.0/rnn_model.pkl",
    "iso_forest.pkl": "https://github.com/Nhlabatsi/phishing_detector/releases/download/v1.0/iso_forest.pkl",
    "tokenizer.pkl": "https://github.com/Nhlabatsi/phishing_detector/releases/download/v1.0/tokenizer.pkl",
    "scaler.pkl": "https://github.com/Nhlabatsi/phishing_detector/releases/download/v1.0/scaler.pkl"
}

# -------------------
# Download utilities
# -------------------
def download_file(url, dest_path):
    """Download a file from URL to destination path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {str(e)}")
        return False

def ensure_models_available():
    """Check if models are available, download if missing"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    all_models_available = True
    
    for model_name, model_url in MODEL_URLS.items():
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            st.info(f"Downloading {model_name}...")
            if download_file(model_url, model_path):
                st.success(f"Downloaded {model_name}")
            else:
                all_models_available = False
    
    return all_models_available

# -------------------
# Load models
# -------------------
@st.cache_resource
def load_models():
    # Ensure models are available
    if not ensure_models_available():
        st.error("Failed to download required models. Please check your internet connection.")
        st.stop()
    
    # Load models
    models_dir = "models"
    try:
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        rnn_model = joblib.load(os.path.join(models_dir, "rnn_model.pkl"))
        iso_forest = joblib.load(os.path.join(models_dir, "iso_forest.pkl"))
        tokenizer = joblib.load(os.path.join(models_dir, "tokenizer.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        return bert_model, rnn_model, iso_forest, tokenizer, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# -------------------
# Preprocessing
# -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_features(text):
    suspicious_words = ["verify", "update", "login", "password", "account", "bank", "urgent", "security", "alert", "suspend"]
    return np.array([
        len(text),
        len(text.split()),
        text.count("URL"),
        text.count("EMAIL"),
        sum(1 for word in suspicious_words if word in text),
    ]).reshape(1, -1)

# -------------------
# Prediction pipeline with explanations
# -------------------
def predict_with_explanation(email_text):
    clean = clean_text(email_text)
    
    # Extract features for explanation
    features = extract_features(clean)
    word_count = features[0][1]
    url_count = features[0][2]
    email_count = features[0][3]
    suspicious_word_count = features[0][4]
    
    # Classic features
    classic_scaled = scaler.transform(features)
    
    # RNN prediction
    seq = tokenizer.texts_to_sequences([clean])
    seq_padded = np.zeros((1, 100))
    if len(seq[0]) > 0:
        seq_padded[0, :min(len(seq[0]), 100)] = seq[0][:100]
    rnn_pred = rnn_model.predict(seq_padded, verbose=0)[0][0]
    
    # Isolation Forest anomaly score
    iso_pred = iso_forest.predict(classic_scaled)[0]
    iso_score = "Anomalous" if iso_pred == -1 else "Normal"
    
    # Final decision (weighted)
    final_score = (rnn_pred + (0.3 if iso_pred == -1 else 0)) / 1.3
    verdict = "Phishing" if final_score > 0.5 else "Legitimate"
    
    # Generate explanations
    explanations = []
    
    # RNN-based explanations
    if rnn_pred > 0.7:
        explanations.append("High linguistic similarity to known phishing emails")
    elif rnn_pred > 0.5:
        explanations.append("Moderate linguistic similarity to suspicious emails")
    else:
        explanations.append("Linguistic patterns appear normal")
    
    # Feature-based explanations
    if url_count > 2:
        explanations.append(f"Contains {url_count} URLs (suspiciously high)")
    elif url_count > 0:
        explanations.append(f"Contains {url_count} URL(s)")
        
    if email_count > 2:
        explanations.append(f"Contains {email_count} email addresses (suspiciously high)")
    elif email_count > 0:
        explanations.append(f"Contains {email_count} email address(es)")
    
    if suspicious_word_count > 3:
        explanations.append(f"Contains {suspicious_word_count} suspicious keywords (high risk)")
    elif suspicious_word_count > 0:
        explanations.append(f"Contains {suspicious_word_count} suspicious keyword(s)")
    
    # Length-based explanations
    if word_count > 150:
        explanations.append("Email is unusually long")
    elif word_count < 20:
        explanations.append("Email is unusually short")
    
    # Isolation Forest explanation
    if iso_pred == -1:
        explanations.append("Metadata patterns are anomalous compared to legitimate emails")
    
    return {
        "rnn_score": rnn_pred,
        "iso_score": iso_score,
        "final_score": final_score,
        "verdict": verdict,
        "explanations": explanations,
        "features": {
            "word_count": word_count,
            "url_count": url_count,
            "email_count": email_count,
            "suspicious_word_count": suspicious_word_count
        }
    }

# -------------------
# Batch processing
# -------------------
def process_batch(df, text_column):
    results = []
    for idx, row in df.iterrows():
        if pd.isna(row[text_column]):
            continue
            
        email_text = str(row[text_column])
        result = predict_with_explanation(email_text)
        
        results.append({
            "Index": idx,
            "Email_Text": email_text[:100] + "..." if len(email_text) > 100 else email_text,
            "Verdict": result["verdict"],
            "Risk_Score": result["final_score"],
            "RNN_Score": result["rnn_score"],
            "Anomaly_Detection": result["iso_score"],
            "Explanations": " | ".join(result["explanations"]),
            "Word_Count": result["features"]["word_count"],
            "URL_Count": result["features"]["url_count"],
            "Email_Count": result["features"]["email_count"],
            "Suspicious_Words": result["features"]["suspicious_word_count"]
        })
    
    return pd.DataFrame(results)

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Batch Phishing Detector", page_icon="✉️", layout="wide")
st.title("✉️ Batch Phishing Email Detector")
st.write("Analyze single emails or upload a CSV file containing multiple emails to detect phishing attempts.")

# Load models (this will trigger download if needed)
try:
    bert_model, rnn_model, iso_forest, tokenizer, scaler = load_models()
    st.success("✅ Models loaded successfully!")
except:
    st.error("❌ Failed to load models. Please check your internet connection.")
    st.stop()

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Single Email", "Upload CSV", "CSV File Path"])

with tab1:
    st.header("Single Email Analysis")
    email_input = st.text_area("Paste your email text here:", height=200)
    
    if st.button("Analyze Single Email"):
        if email_input.strip():
            result = predict_with_explanation(email_input)
            
            st.subheader("🔎 Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Verdict", result["verdict"], 
                         delta="High Risk" if result["verdict"] == "Phishing" else "Low Risk",
                         delta_color="inverse" if result["verdict"] == "Phishing" else "normal")
            with col2:
                st.metric("Risk Score", f"{result['final_score']:.3f}")
            with col3:
                st.metric("RNN Score", f"{result['rnn_score']:.3f}")
            with col4:
                st.metric("Anomaly", result["iso_score"])
            
            st.subheader("📋 Explanations")
            for explanation in result["explanations"]:
                st.write(f"• {explanation}")
                
            st.subheader("📊 Features")
            feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
            with feat_col1:
                st.metric("Word Count", result["features"]["word_count"])
            with feat_col2:
                st.metric("URL Count", result["features"]["url_count"])
            with feat_col3:
                st.metric("Email Count", result["features"]["email_count"])
            with feat_col4:
                st.metric("Suspicious Words", result["features"]["suspicious_word_count"])
                
        else:
            st.warning("Please paste an email first.")

with tab2:
    st.header("CSV File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Let user select which column contains the email text
            text_column = st.selectbox("Select the column containing email text", df.columns)
            
            if st.button("Analyze Batch"):
                with st.spinner("Analyzing emails..."):
                    results_df = process_batch(df, text_column)
                
                st.subheader("Analysis Results")
                
                # Summary statistics
                phishing_count = len(results_df[results_df["Verdict"] == "Phishing"])
                total_count = len(results_df)
                st.metric("Phishing Emails Detected", f"{phishing_count}/{total_count}",
                         delta=f"{(phishing_count/total_count*100):.1f}%",
                         delta_color="inverse" if phishing_count > 0 else "normal")
                
                # Display results table
                st.dataframe(results_df, use_container_width=True)
                
                # Download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="phishing_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("CSV File Path Input")
    csv_path = st.text_input("Enter the full path to your CSV file:")
    
    if csv_path:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                st.success("File found and loaded successfully!")
                
                # Let user select which column contains the email text
                text_column = st.selectbox("Select the column containing email text", df.columns, key="path_column")
                
                if st.button("Analyze from Path"):
                    with st.spinner("Analyzing emails..."):
                        results_df = process_batch(df, text_column)
                    
                    st.subheader("Analysis Results")
                    
                    # Summary statistics
                    phishing_count = len(results_df[results_df["Verdict"] == "Phishing"])
                    total_count = len(results_df)
                    st.metric("Phishing Emails Detected", f"{phishing_count}/{total_count}",
                             delta=f"{(phishing_count/total_count*100):.1f}%",
                             delta_color="inverse" if phishing_count > 0 else "normal")
                    
                    # Display results table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="phishing_analysis_results.csv",
                        mime="text/csv",
                        key="path_download"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.error("File not found. Please check the path and try again.")


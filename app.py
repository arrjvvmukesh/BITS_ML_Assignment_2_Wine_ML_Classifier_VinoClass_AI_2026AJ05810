import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Cosmetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VinoClass | Premium Wine Quality Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Burgundy & Warm Gold Custom Styling
st.markdown("""
<style>
    /* Main Background Accent */
    .stApp {
        background-color: #0f0e13;
        color: #e6e6e6;
    }
    
    /* Custom Header Styling */
    .header-title {
        color: #722F37;
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .header-subtitle {
        color: #c9a050;
        font-size: 1.1rem;
        font-style: italic;
        margin-bottom: 25px;
    }
    
    /* Card Component */
    .metric-card {
        background: linear-gradient(135deg, #1c1921 0%, #291a24 100%);
        border: 1px solid #4a1f28;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .metric-card h4 {
        color: #c9a050;
        margin: 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 5px 0 0 0;
    }

    /* Streamlit Metric Box Override */
    div[data-testid="stMetricValue"] {
        color: #c9a050 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Header & Title Section
# -----------------------------------------------------------------------------
st.markdown('<div class="header-title">🍷 VinoClass AI</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Physicochemical Wine Quality Analytics & Machine Learning Benchmarks</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Developed By: APPANA MUKESH </div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle"> BITS ID: 2026AJ05910 </div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# 3. Model & Preprocessor Loader Function
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_dir = "model"
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    
    models = {
        "Decision Tree": os.path.join(model_dir, "decision_tree.joblib"),
        "Support Vector Machine": os.path.join(model_dir, "svm.joblib"),
        "Logistic Regression": os.path.join(model_dir, "logistic_regression.joblib"),
        "Naive Bayes": os.path.join(model_dir, "naive_bayes.joblib"),
        "KNN": os.path.join(model_dir, "knn.joblib")
    }
    
    loaded_models = {}
    preprocessor = None
    
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    
    for name, path in models.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
            
    return preprocessor, loaded_models

preprocessor, models_dict = load_artifacts()

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=500&auto=format&fit=crop", use_container_width=True)
st.sidebar.header("⚙️ Control Panel")

if not models_dict or preprocessor is None:
    st.error("⚠️ Model artifacts missing! Run `train_and_export.py` first to create the `model/` directory artifacts.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select Model for Individual Focus:",
    options=list(models_dict.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Target Label Key:**
- **`1` (High Quality):** Quality rating ≥ 7
- **`0` (Standard Quality):** Quality rating < 7
""")

# -----------------------------------------------------------------------------
# 5. Data Input & Inference Section
# -----------------------------------------------------------------------------
st.subheader("📁 Upload Test Dataset")
uploaded_file = st.file_uploader("Upload `test_data.csv` for evaluation:", type=["csv"])

# Option to use default saved test data if present
if uploaded_file is None:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
        st.info("ℹ️ Using default local `test_data.csv` file for evaluation.")
    else:
        st.warning("Please upload a `test_data.csv` file to begin model evaluation.")
        st.stop()
else:
    test_df = pd.read_csv(uploaded_file)

# Validate dataset columns
if "target" not in test_df.columns:
    st.error("Uploaded CSV must contain a target column named `'target'`.")
    st.stop()

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Transform features
X_scaled = preprocessor.transform(X_test)
active_model = models_dict[selected_model_name]

# Active Model Predictions
y_pred = active_model.predict(X_scaled)
if hasattr(active_model, "predict_proba"):
    y_proba = active_model.predict_proba(X_scaled)[:, 1]
else:
    y_proba = active_model.decision_function(X_scaled)

# -----------------------------------------------------------------------------
# 6. Active Model Metric Scorecards
# -----------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

st.markdown(f"### 📊 Active Model Performance: `{selected_model_name}`")

m1, m2, m3, m4, m5, m6 = st.columns(6)
metrics_data = [
    ("Accuracy", f"{acc:.4f}"),
    ("AUC Score", f"{auc:.4f}"),
    ("Precision", f"{prec:.4f}"),
    ("Recall", f"{rec:.4f}"),
    ("F1-Score", f"{f1:.4f}"),
    ("MCC", f"{mcc:.4f}")
]

for col, (label, val) in zip([m1, m2, m3, m4, m5, m6], metrics_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <p>{val}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Multi-Tab Analytics Section (Including Comparative Benchmark)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏆 Comparative Benchmark", 
    "📉 Confusion Matrix", 
    "📋 Predictions Preview"
])

# -----------------------------------------------------------------------------
# TAB 1: Comparative Benchmark Across All 5 Models
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🥇 5-Model Benchmark Comparison")
    
    # Calculate metrics for all loaded models
    benchmark_results = []
    for model_name, model_obj in models_dict.items():
        m_pred = model_obj.predict(X_scaled)
        if hasattr(model_obj, "predict_proba"):
            m_proba = model_obj.predict_proba(X_scaled)[:, 1]
        else:
            m_proba = model_obj.decision_function(X_scaled)
            
        benchmark_results.append({
            "ML Model Name": model_name,
            "Accuracy": round(accuracy_score(y_test, m_pred), 4),
            "AUC": round(roc_auc_score(y_test, m_proba), 4),
            "Precision": round(precision_score(y_test, m_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, m_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, m_pred, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_test, m_pred), 4)
        })
    
    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df = benchmark_df.sort_values(by="F1", ascending=False).reset_index(drop=True)
    
    # Display styled interactive table highlighting maximum values in each metric
    st.dataframe(
        benchmark_df.style.highlight_max(
            axis=0, 
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], 
            color="#4a1f28"
        ).format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True
    )
    
    # Comparative Bar Chart
    st.markdown("#### 📊 Metric Comparison Visualization")
    
    fig_bm, ax_bm = plt.subplots(figsize=(10, 4))
    fig_bm.patch.set_facecolor('#0f0e13')
    ax_bm.set_facecolor('#0f0e13')
    
    # Plot Accuracy vs F1 vs AUC
    melted_df = benchmark_df.melt(id_vars=["ML Model Name"], value_vars=["Accuracy", "AUC", "F1"], var_name="Metric", value_name="Score")
    
    sns.barplot(
        data=melted_df, 
        x="ML Model Name", 
        y="Score", 
        hue="Metric", 
        palette=["#722F37", "#c9a050", "#4a1f28"],
        ax=ax_bm
    )
    
    ax_bm.set_ylim(0, 1.05)
    ax_bm.set_ylabel("Score", color="#c9a050", fontsize=11)
    ax_bm.set_xlabel("", color="#c9a050")
    ax_bm.tick_params(colors="white")
    ax_bm.legend(facecolor='#1c1921', edgecolor='#4a1f28', labelcolor='white')
    
    st.pyplot(fig_bm)

# -----------------------------------------------------------------------------
# TAB 2: Active Model Confusion Matrix
# -----------------------------------------------------------------------------
with tab2:
    st.markdown(f"### 📉 Confusion Matrix: `{selected_model_name}`")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    fig_cm.patch.set_facecolor('#0f0e13')
    ax_cm.set_facecolor('#0f0e13')
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="YlOrRd", 
        cbar=False,
        xticklabels=["Standard (0)", "High (1)"],
        yticklabels=["Standard (0)", "High (1)"],
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted Quality Label", color="#c9a050", fontsize=11)
    ax_cm.set_ylabel("Actual Quality Label", color="#c9a050", fontsize=11)
    ax_cm.tick_params(colors="white")
    
    st.pyplot(fig_cm)

# -----------------------------------------------------------------------------
# TAB 3: Batch Prediction Table Preview
# -----------------------------------------------------------------------------
with tab3:
    st.markdown(f"### 📋 Dataset Predictions Preview: `{selected_model_name}`")
    results_preview = X_test.copy()
    results_preview["Actual Target"] = y_test
    results_preview["Predicted Label"] = y_pred
    results_preview["Positive Class Proba"] = np.round(y_proba, 4)
    
    st.dataframe(results_preview.head(25), use_container_width=True)
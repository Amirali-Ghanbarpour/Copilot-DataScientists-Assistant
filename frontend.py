import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu 
from Backend import pearson_correlation, spearman_correlation_heatmap, spearman_correlation, pearson_correlation_heatmap
from scipy.stats import gaussian_kde
from LLM import eda_scan, eda_scan_each_feature, model_recommendation, ai_answer
from preprocessing import set_session_state, preprocessing_before_training_models, divide_the_dataset_into_x_y, one_hot_encoding, most_frequent_missing_data_handeling, robust_scaler_for_numerical_features, median_imputer_missing_data_handling, select_feature_type_categorical, select_feature_type_numerical
from sklearn.model_selection import train_test_split
from modeling import train_testsplit, report_for_linear_models, metrics, linear_models, classification_rep, confusion_matrixx, logistic_regression, random_forest_calssifer, svc_classifier, knn_classifier
from sklearn.preprocessing import LabelEncoder
import textwrap


# ============================================================
# FRONTEND STYLE
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* =========================================================
   GLOBAL
   ========================================================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F5F7FB;
    color: #172B4D;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}


/* =========================================================
   HEADER
   ========================================================= */

.main-title {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 28px 34px 25px 34px;
    margin-bottom: 30px;

    border: 1px solid #E7EBF2;

    box-shadow:
        0 8px 25px rgba(31, 50, 81, 0.07);

    text-align: center;
}

.main-title h1 {
    color: #172B4D;
    font-size: 2.35rem;
    font-weight: 700;
    margin: 0 0 12px 0;
    letter-spacing: -0.7px;
}

.main-title em {
    color: #65748B;
    font-size: 1.05rem;
    font-style: italic;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    color: #172B4D;
    font-size: 1.45rem;
    font-weight: 650;
    margin-bottom: 4px;
}

.section-description {
    color: #718096;
    font-size: 0.92rem;
    margin-bottom: 18px;
}


/* =========================================================
   UPLOAD AREA
   ========================================================= */

.upload-card {
    background: #FFFFFF;
    border: 1px solid #E3E8F0;
    border-radius: 18px;

    padding: 25px 28px;

    margin-bottom: 26px;

    box-shadow:
        0 5px 18px rgba(31, 50, 81, 0.055);
}

[data-testid="stFileUploader"] {
    background: #F8FAFD;
    border: 2px dashed #B9CBE7;
    border-radius: 14px;
    padding: 10px 14px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #4A90E2;
    background: #F5F9FF;
}


/* =========================================================
   NAVIGATION
   ========================================================= */

div[data-testid="stHorizontalBlock"] {
    gap: 0.8rem;
}

.nav-container {
    background: #FFFFFF;
    border: 1px solid #DDE4EF;
    border-radius: 14px;
    padding: 5px;
    margin-bottom: 28px;
}


/* =========================================================
   DATASET PREVIEW
   ========================================================= */

.preview-card {
    background: #FFFFFF;
    border: 1px solid #E1E6EE;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;

    box-shadow:
        0 5px 18px rgba(31, 50, 81, 0.05);
}


/* Streamlit dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #E1E6EE;
    border-radius: 14px;
    overflow: hidden;
}


/* =========================================================
   INFORMATION CARDS
   ========================================================= */

.info-card {
    background: #FFFFFF;

    border: 1px solid #E1E6EE;
    border-radius: 16px;

    padding: 20px;

    min-height: 125px;

    box-shadow:
        0 5px 16px rgba(31, 50, 81, 0.055);
}

.info-card-title {
    color: #263A5A;
    font-size: 0.95rem;
    font-weight: 650;
    margin-bottom: 10px;
}

.info-card-value {
    color: #6B7C93;
    font-size: 0.9rem;
}

.info-card-icon {
    font-size: 1.8rem;
    margin-bottom: 5px;
}


/* =========================================================
   STATISTICS CARDS
   ========================================================= */

.stat-card {
    background: #FFFFFF;

    border: 1px solid #E1E6EE;
    border-radius: 16px;

    padding: 18px 20px;

    min-height: 110px;

    box-shadow:
        0 5px 16px rgba(31, 50, 81, 0.05);
}

.stat-title {
    color: #354A6A;
    font-size: 0.95rem;
    font-weight: 650;
}

.stat-feature {
    color: #718096;
    font-size: 0.83rem;
    margin-top: 5px;
}

.stat-value {
    color: #172B4D;
    font-size: 1.15rem;
    font-weight: 650;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1, h2, h3 {
    color: #172B4D !important;
}

h2 {
    font-weight: 650 !important;
}

h3 {
    font-weight: 600 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: #2F80ED;
    color: white;

    border: none;
    border-radius: 10px;

    padding: 0.55rem 1.35rem;

    font-weight: 600;

    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #256FD1;
    color: white;

    box-shadow: 0 5px 14px rgba(47, 128, 237, 0.25);
}


/* =========================================================
   SELECTBOX / RADIO / INPUTS
   ========================================================= */

div[data-baseweb="select"] {
    border-radius: 10px;
}

.stTextInput input {
    border-radius: 10px;
}

.stSelectbox label,
.stRadio label {
    color: #354A6A;
    font-weight: 600;
}


/* =========================================================
   TABLES
   ========================================================= */

[data-testid="stTable"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #E1E6EE;
}

[data-testid="stTable"] table {
    background: white;
}

[data-testid="stTable"] thead tr {
    background: #EEF4FC;
}

[data-testid="stTable"] thead th {
    color: #315A92;
    font-weight: 650;
}

[data-testid="stTable"] tbody tr:nth-child(even) {
    background: #FAFBFD;
}


/* =========================================================
   CHAT
   ========================================================= */

[data-testid="stChatMessage"] {
    border-radius: 14px;
}


/* =========================================================
   EXPANDERS
   ========================================================= */

[data-testid="stExpander"] {
    border-radius: 14px;
    border: 1px solid #E1E6EE;
    background: #FFFFFF;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border: none;
    border-top: 1px solid #E5EAF1;
    margin: 25px 0;
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-thumb {
    background: #C4CFDE;
    border-radius: 20px;
}

::-webkit-scrollbar-track {
    background: #F1F4F8;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E1E6EE;
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 5px 16px rgba(31, 50, 81, 0.055);
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 7px 20px rgba(31, 50, 81, 0.08);
}

[data-testid="stVerticalBlockBorderWrapper"] h3 {
    font-size: 1rem !important;
    margin-bottom: 12px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] .stCaption {
    color: #718096;
}

[data-testid="stVerticalBlockBorderWrapper"] strong {
    font-size: 1.2rem;
    color: #172B4D;
}

/* =========================================================
   CORRELATION FIELDS
   ========================================================= */

.correlation-section {
    margin-top: 10px;
    margin-bottom: 22px;
}

.correlation-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.correlation-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 16px;
}

.correlation-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.correlation-card-title {
    font-size: 1.05rem;
    font-weight: 650;
    color: #243B5A;
    margin-bottom: 14px;
}

.correlation-card-subtitle {
    font-size: 0.84rem;
    color: #718096;
    margin-bottom: 16px;
}

.missing-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.missing-card-title {
    font-size: 1.05rem;
    font-weight: 650;
    color: #243B5A;
    margin-bottom: 8px;
}

.metric-number {
    font-size: 1.7rem;
    font-weight: 700;
    color: #2563EB;
}

.description-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #172B4D;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* =========================================================
   DISTRIBUTIONS
   ========================================================= */

.distribution-section {
    margin-top: 10px;
    margin-bottom: 24px;
}

.distribution-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.distribution-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

.distribution-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 18px 20px 10px 20px;
    margin-bottom: 24px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.distribution-card-title {
    font-size: 1rem;
    font-weight: 650;
    color: #243B5A;
    margin-bottom: 4px;
}

.distribution-card-subtitle {
    color: #8A96A8;
    font-size: 0.8rem;
    margin-bottom: 10px;
}

/* Plot container */

[data-testid="stImage"] img {
    border-radius: 12px;
}

/* Give matplotlib containers a little breathing room */

[data-testid="stPyplot"] {
    padding-top: 5px;
    padding-bottom: 5px;
}

/* =========================================================
   OUTLIERS DETECTION
   ========================================================= */

.outlier-section {
    margin-top: 10px;
    margin-bottom: 25px;
}

.outlier-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.outlier-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

.outlier-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 22px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.outlier-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #243B5A;
    margin-bottom: 5px;
}

.outlier-count {
    display: inline-block;
    background: #EEF4FF;
    color: #2563EB;
    border-radius: 10px;
    padding: 5px 10px;
    font-size: 0.82rem;
    font-weight: 650;
    margin-bottom: 16px;
}

.outlier-count-danger {
    background: #FFF1F2;
    color: #DC2626;
}

.outlier-count-success {
    background: #ECFDF5;
    color: #059669;
}

.outlier-boundary {
    color: #718096;
    font-size: 0.82rem;
    margin-top: 5px;
}

.outlier-values-title {
    font-size: 0.95rem;
    font-weight: 650;
    color: #354A6A;
    margin-top: 18px;
    margin-bottom: 8px;
}

.outlier-empty {
    color: #059669;
    font-size: 0.9rem;
    background: #ECFDF5;
    border-radius: 10px;
    padding: 10px 14px;
}

/* =========================================================
   FEATURE ANALYSIS
   ========================================================= */

.feature-section {
    margin-top: 10px;
    margin-bottom: 25px;
}

.feature-section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.feature-section-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

.feature-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 18px 20px 10px 20px;
    margin-bottom: 22px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.feature-card-title {
    font-size: 1rem;
    font-weight: 650;
    color: #243B5A;
    margin-bottom: 4px;
}

.feature-card-subtitle {
    color: #8A96A8;
    font-size: 0.8rem;
    margin-bottom: 10px;
}

.scatter-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 22px;
    margin-top: 10px;
    margin-bottom: 24px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.scatter-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #243B5A;
    margin-bottom: 5px;
}

.scatter-card-description {
    color: #718096;
    font-size: 0.85rem;
    margin-bottom: 18px;
}

.scatter-selection-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #354A6A;
    margin-bottom: 5px;
}

[data-testid="stPyplot"] {
    padding-top: 5px;
    padding-bottom: 5px;
}

/* =========================================================
   LLM ANALYSIS
   ========================================================= */

.llm-section {
    margin-top: 10px;
    margin-bottom: 25px;
}

.llm-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.llm-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 25px;
}

.llm-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 22px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.llm-card:hover {
    box-shadow: 0 7px 22px rgba(31, 50, 81, 0.08);
}

.llm-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #243B5A;
    margin-bottom: 6px;
}

.llm-card-description {
    color: #718096;
    font-size: 0.84rem;
    line-height: 1.5;
    margin-bottom: 18px;
}

.llm-icon {
    font-size: 1.5rem;
    margin-bottom: 8px;
}

.llm-result {
    background: #F8FAFD;
    border: 1px solid #E5EAF1;
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
    color: #354A6A;
    line-height: 1.7;
}

.llm-result-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 10px;
}

.llm-status {
    background: #EEF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 10px 14px;
    color: #355070;
    font-size: 0.85rem;
    margin-top: 12px;
}

/* LLM buttons */

.llm-card .stButton > button {
    width: 100%;
    border-radius: 11px;
    padding: 0.6rem 1rem;
    font-weight: 650;
}

/* Different visual hierarchy for the cards */

.llm-card-primary {
    border-top: 4px solid #2563EB;
}

.llm-card-secondary {
    border-top: 4px solid #38BDF8;
}

.llm-card-tertiary {
    border-top: 4px solid #6366F1;
}

/* =========================================================
   CHAT WITH AI
   ========================================================= */

.chat-section {
    margin-top: 10px;
    margin-bottom: 20px;
}

.chat-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.chat-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 22px;
}

.chat-info-card {
    background: linear-gradient(135deg, #EEF5FF, #F7FBFF);
    border: 1px solid #DCE9FA;
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 25px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.04);
}

.chat-info-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

.chat-info-title {
    font-size: 1rem;
    font-weight: 700;
    color: #243B5A;
    margin-bottom: 5px;
}

.chat-info-text {
    color: #718096;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #EEF4FF;
    border: 1px solid #DCE8FA;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    box-shadow: 0 4px 14px rgba(31, 50, 81, 0.045);
}

/* Chat input */

[data-testid="stChatInput"] {
    border-radius: 16px;
}

[data-testid="stChatInput"] textarea {
    border-radius: 14px;
}

/* Empty chat state */

.chat-empty {
    text-align: center;
    padding: 35px 20px;
    color: #718096;
}

.chat-empty-icon {
    font-size: 2.2rem;
    margin-bottom: 10px;
}

.chat-empty-title {
    font-size: 1rem;
    font-weight: 650;
    color: #354A6A;
    margin-bottom: 5px;
}

.chat-empty-text {
    font-size: 0.85rem;
}

/* =========================================================
   AI ASSISTANT / MACHINE LEARNING
   ========================================================= */

.ml-section {
    margin-top: 10px;
    margin-bottom: 25px;
}

.ml-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 5px;
}

.ml-description {
    color: #718096;
    font-size: 0.9rem;
    margin-bottom: 25px;
}

/* Target / problem selection */

.ml-control-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 20px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

.ml-control-title {
    font-size: 1rem;
    font-weight: 700;
    color: #243B5A;
    margin-bottom: 5px;
}

.ml-control-description {
    color: #718096;
    font-size: 0.84rem;
    margin-bottom: 15px;
}

/* Run ML */

.ml-run-card {
    background: linear-gradient(135deg, #EEF5FF, #F7FBFF);
    border: 1px solid #DCE9FA;
    border-radius: 18px;
    padding: 20px 22px;
    margin: 10px 0 25px 0;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.04);
}

.ml-run-title {
    font-size: 1rem;
    font-weight: 700;
    color: #243B5A;
    margin-bottom: 5px;
}

.ml-run-description {
    color: #718096;
    font-size: 0.84rem;
    margin-bottom: 15px;
}

.ml-run-card .stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.65rem 1rem;
}

/* Results */

.ml-results-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #172B4D;
    margin: 25px 0 5px 0;
}

.ml-results-description {
    color: #718096;
    font-size: 0.85rem;
    margin-bottom: 15px;
}

/* Result table */

.ml-results-table {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 15px;
    margin-top: 10px;
    box-shadow: 0 5px 18px rgba(31, 50, 81, 0.055);
}

/* Target badge */

.ml-target-badge {
    display: inline-block;
    background: #EEF4FF;
    color: #2563EB;
    border-radius: 10px;
    padding: 6px 12px;
    font-size: 0.82rem;
    font-weight: 650;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <h1>🤖 Copilot Datascientist Assistant</h1>
        <em>Phase 1 — Upload & Preview your dataset</em>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD SECTION
# ============================================================


st.markdown(
    """
    <div class="section-title">
        Upload your dataset
    </div>

    <div class="section-description">
        Supported formats: CSV or Excel
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_data = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    label_visibility="collapsed"
)


# the data manipulations on the dataset
eda_results = {}
if uploaded_data:
    uploaded_data = pd.read_csv(uploaded_data)
    numerical_features = select_feature_type_numerical(uploaded_data)
    categorical_features = select_feature_type_categorical(uploaded_data)

    #calculating IQR for dictionary input:
    list_of_features = uploaded_data.columns
    dict_of_outliers_per_feature = {}
    for feature in numerical_features:
        Q1 = uploaded_data[feature].quantile(0.25)
        Q3 = uploaded_data[feature].quantile(0.75)
        IQR = Q3-Q1
        lower_boundry = Q1 - 1.5 * IQR
        upper_boundry = Q3 + 1.5 * IQR
        outliers = uploaded_data[(uploaded_data[feature] < lower_boundry) | (uploaded_data[feature] > upper_boundry)]
        index_of_outliers = outliers.index
        index_of_the_outliers_as_list = index_of_outliers.to_list()
        dict_of_outliers_per_feature[feature] = index_of_the_outliers_as_list

    # Global dictionary as LLM input
    eda_results["stats"] = uploaded_data.describe().to_dict()
    eda_results["missing values"] = uploaded_data.isnull().sum().to_dict()
    
    #dict for numerical stats
    numeric_sats = {}
    pearson_cor = {}
    spearman_cor = {}
    for feature in numerical_features:
        eda_results[feature] = {}
        numeric_sats[feature] = {
        "Min" : uploaded_data[feature].min(),
        "Max" : uploaded_data[feature].max(),
        "Mean" : uploaded_data[feature].mean(),        
        }

        for other_feature in numerical_features:
            pearson_cor[other_feature] = uploaded_data[feature].corr(
                uploaded_data[other_feature],
                method = 'pearson'
            )
            spearman_cor[other_feature] = uploaded_data[feature].corr(
                uploaded_data[other_feature],
                method = 'spearman'
            )
        eda_results[feature]["Pearson Correlation"] = pearson_cor
        eda_results[feature]["Spearman Correlation"] = spearman_cor



    eda_results["Outliers"] = dict_of_outliers_per_feature
    #--------------Phase 3 part 2-------------------------------------------
    each_feature_summary = {}
    #Note: all the int,float outputs needed to be wrapped inside a float(), int() functions because Json file only accepts data from dictionary which are float64,int64
    for feature in list_of_features:
        each_feature_summary[feature] = {}
        each_feature_summary[feature]["Feature Discription"] : uploaded_data[feature].describe().to_dict()
        each_feature_summary[feature]["Missing Values"] : int(uploaded_data[feature].isnull().sum())

        if feature in numerical_features:    
            each_feature_summary[feature]["Min"] : float(uploaded_data[feature].min())
            each_feature_summary[feature]["Max"] : float(uploaded_data[feature].max())
            each_feature_summary[feature]["Mean:"] : float(uploaded_data[feature].mean())
            each_feature_summary[feature]["Median:"] : float(uploaded_data[feature].median())
            each_feature_summary[feature]["Standard Deviation"] : float(uploaded_data[feature].std())
            each_feature_summary[feature]["Outliers"] : dict_of_outliers_per_feature[feature]
            
            each_feature_summary[feature]["Pearson Correlation"] = {
                other_feature : uploaded_data[feature].corr(uploaded_data[other_feature], method = 'pearson')
                for other_feature in numerical_features
            }

            each_feature_summary[feature]["Spearman Correlation"] = {
                other_feature : uploaded_data[feature].corr(uploaded_data[other_feature] , method='spearman')
                for other_feature in numerical_features
            }
    #st.write(each_feature_summary)
    #-----------------------------------------------------------------
    #Option Menu 
    selected = option_menu(
        menu_title = None,
        options=["Developing Stats" , "Correlation Fields" , "Distributions" , "Outliers Detection" , "Feature Analysis" , "LLM" , "Chat with AI" , "AI Assistant"],
        orientation = "horizontal",
        styles={
        "container": {"padding": "5px", "background-color": "#FFFFFF" , "border": "1px solid #DDE4EF", "border-radius": "14px", "margin-bottom": "28px",},
        "icon": {"font-size": "0px"}, 
        "nav-link": {"font-size": "13px", "text-align": "center", "font-family":"Inter, sans-serif" , "font-weight": "500", "margin":"2px",  "padding": "10px 12px", "border-radius": "9px", "--hover-color": "#EEF4FC" , "color": "#42526E"},
        "nav-link-selected": {"background-color": "#2F80ED" , "color": "#FFFFFF" , "font-weight": "600"},
    }
        



    )

    if selected == "Developing Stats":
        
    # ========================================================
    # DATASET PREVIEW
    # ========================================================

        st.markdown(
            """
            <div class="section-title">
                Dataset Preview
            </div>

            <div class="section-description">
                First 5 rows of your Dataset:
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            uploaded_data.head(5),
            use_container_width=True,
            hide_index=False
        )


        # ========================================================
        # DATASET INFORMATION CARDS
        # ========================================================

        rows, columns = uploaded_data.shape
        missing_mask = uploaded_data.isnull().sum()

        st.write("")

        column1, column2, column3 = st.columns(3)

        with column1:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-icon">📄</div>
                    <div class="info-card-title">Shape</div>
                    <div class="info-card-value">
                        Rows: <b>{rows}</b><br>
                        Columns: <b>{columns}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with column2:

            total_missing = int(missing_mask.sum())

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-icon">⚠️</div>
                    <div class="info-card-title">Missing Values</div>
                    <div class="info-card-value">
                        Total missing values:
                        <b>{total_missing}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with column3:

            numerical_count = len(numerical_features)
            categorical_count = len(categorical_features)

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-icon">📊</div>
                    <div class="info-card-title">Column Info</div>
                    <div class="info-card-value">
                        Numerical: <b>{numerical_count}</b><br>
                        Categorical: <b>{categorical_count}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ========================================================
        # COLUMN INFORMATION
        # ========================================================

        st.write("")
        st.markdown(
            """
            <div class="section-title">
                Column Information
            </div>
            """,
            unsafe_allow_html=True
        )

        column_info = pd.DataFrame({
            "Column": uploaded_data.columns,
            "Data Type": uploaded_data.dtypes.astype(str).values,
            "Missing Values": uploaded_data.isnull().sum().values,
            "Non-Null Values": uploaded_data.notnull().sum().values
        })

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )


        # ========================================================
        # BASIC STATS
        # ========================================================
        
        st.write("")

        st.markdown(
            """
            <div class="section-title">
                Basic Stats
            </div>
            """,
            unsafe_allow_html=True
        )

        if len(numerical_features) > 0:

            for start in range(0, len(numerical_features), 3):

                current_features = numerical_features[start:start + 3]

                stat_columns = st.columns(3)

                for column, feature in zip(stat_columns, current_features):

                    with column:

                        feature_mean = uploaded_data[feature].mean()
                        feature_max = uploaded_data[feature].max()
                        feature_min = uploaded_data[feature].min()

                        st.html(
                            f"""
                            <div class="stat-card">

                                <div class="stat-title">
                                    📊 {feature}
                                </div>

                                <div class="stat-feature">
                                    Mean
                                </div>

                                <div class="stat-value">
                                    {feature_mean:.2f}
                                </div>

                                <div class="stat-feature">
                                    Min: {feature_min:.2f}
                                    &nbsp;&nbsp;|&nbsp;&nbsp;
                                    Max: {feature_max:.2f}
                                </div>

                            </div>
                            """
                        )

                st.write("")


    
    if selected == "Correlation Fields":

        # ========================================================
        # CORRELATION ANALYSIS
        # ========================================================

        numerical_df = uploaded_data.select_dtypes(
            include=['int64', 'float64']
        )

        # ========================================================
        # PEARSON CORRELATION
        # ========================================================

        st.html(
            """
            <div class="correlation-section">

                <div class="correlation-title">
                    Pearson Correlation
                </div>

                <div class="correlation-description">
                    Measures the linear relationship between numerical features.
                </div>

            </div>
            """
        )

        pearson_matrix = numerical_df.corr(method='pearson')

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    📊 Pearson Correlation Matrix
                </div>

                <div class="correlation-card-subtitle">
                    Values range from -1 to +1. Values closer to ±1 indicate
                    stronger linear relationships.
                </div>

            </div>
            """
        )

        st.dataframe(
            pearson_matrix,
            use_container_width=True
        )


        # ========================================================
        # PEARSON HEATMAP
        # ========================================================

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    🔥 Pearson Correlation Heatmap
                </div>

                <div class="correlation-card-subtitle">
                    Visual representation of linear correlations between features.
                </div>

            </div>
            """
        )

        pearson_corr_heatm = pearson_correlation_heatmap(uploaded_data)

        st.pyplot(
            pearson_corr_heatm,
            use_container_width=True
        )


        # ========================================================
        # SPEARMAN CORRELATION
        # ========================================================

        st.html(
            """
            <div class="correlation-section">

                <div class="correlation-title">
                    Spearman Correlation
                </div>

                <div class="correlation-description">
                    Measures the strength of monotonic relationships between
                    numerical features.
                </div>

            </div>
            """
        )

        spearman_matrix = numerical_df.corr(method='spearman')

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    📈 Spearman Correlation Matrix
                </div>

                <div class="correlation-card-subtitle">
                    Rank-based correlation that can capture monotonic relationships.
                </div>

            </div>
            """
        )

        st.dataframe(
            spearman_matrix,
            use_container_width=True
        )


        # ========================================================
        # SPEARMAN HEATMAP
        # ========================================================

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    🌡️ Spearman Correlation Heatmap
                </div>

                <div class="correlation-card-subtitle">
                    Visual representation of rank-based correlations.
                </div>

            </div>
            """
        )

        spearman_corr_heatm = spearman_correlation_heatmap(uploaded_data)

        st.pyplot(
            spearman_corr_heatm,
            use_container_width=True
        )


        # ========================================================
        # MISSING VALUES
        # ========================================================

        st.html(
            """
            <div class="correlation-section">

                <div class="correlation-title">
                    Missing Values
                </div>

                <div class="correlation-description">
                    Overview of missing observations in each dataset feature.
                </div>

            </div>
            """
        )

        uploaded_data = pd.DataFrame(uploaded_data)

        missing_data_in_dataset = uploaded_data.isnull().sum()

        missing_percentage = (
            missing_data_in_dataset / len(uploaded_data)
        ) * 100


        # --------------------------------------------------------
        # Missing values by number
        # --------------------------------------------------------

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    🔎 Missing Values by Feature
                </div>

            </div>
            """
        )

        missing_values_df = pd.DataFrame({
            "Feature": missing_data_in_dataset.index,
            "Missing Values": missing_data_in_dataset.values
        })

        st.dataframe(
            missing_values_df,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------------
        # Missing percentage
        # --------------------------------------------------------

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    📉 Missing Values Percentage
                </div>

            </div>
            """
        )

        missing_percentage_df = pd.DataFrame({
            "Feature": missing_percentage.index,
            "Missing Percentage (%)": missing_percentage.values
        })

        st.dataframe(
            missing_percentage_df,
            use_container_width=True,
            hide_index=True
        )


        # ========================================================
        # MISSING VALUES BAR CHART
        # ========================================================

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    📊 Missing Values Distribution
                </div>

                <div class="correlation-card-subtitle">
                    Number of missing observations for each feature.
                </div>

            </div>
            """
        )

        list_of_features = uploaded_data.columns

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        ax2.bar(
            list_of_features,
            missing_data_in_dataset
        )

        ax2.set_xlabel("Features")
        ax2.set_ylabel("Missing Values")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )


        # ========================================================
        # DATASET DESCRIPTION
        # ========================================================

        st.html(
            """
            <div class="description-title">
                Dataset Description
            </div>
            """
        )

        st.html(
            """
            <div class="correlation-card">

                <div class="correlation-card-title">
                    📋 Statistical Description
                </div>

                <div class="correlation-card-subtitle">
                    Descriptive statistics for the numerical features.
                </div>

            </div>
            """
        )

        st.dataframe(
            uploaded_data.describe(),
            use_container_width=True
        )



    if selected == "Distributions":
        
        # ========================================================
        # DISTRIBUTIONS
        # ========================================================

        st.html(
            """
            <div class="distribution-section">

                <div class="distribution-title">
                    📊 Distribution Analysis
                </div>

                <div class="distribution-description">
                    Explore the distribution and density of the features
                    in your dataset.
                </div>

            </div>
            """
        )


        # ========================================================
        # HISTOGRAMS
        # ========================================================

        st.html(
            """
            <div class="distribution-section">

                <div class="distribution-title">
                    📊 Histograms
                </div>

                <div class="distribution-description">
                    Histograms show the frequency distribution of each feature.
                </div>

            </div>
            """
        )

        # Histogram
        list_of_features = uploaded_data.columns

        for start in range(0, len(list_of_features), 2):

            current_features = list_of_features[start:start + 2]

            plot_columns = st.columns(2)

            for column, feature in zip(plot_columns, current_features):

                with column:

                    st.html(
                        f"""
                        <div class="distribution-card">

                            <div class="distribution-card-title">
                                📈 {feature}
                            </div>

                            <div class="distribution-card-subtitle">
                                Histogram
                            </div>

                        </div>
                        """
                    )

                    fig3, ax = plt.subplots(figsize=(6, 4))

                    sns.histplot(
                        uploaded_data[feature],
                        kde=False,
                        color='red',
                        bins=30,
                        ax=ax
                    )

                    ax.set_title(
                        f"{feature} Distribution",
                        fontsize=12,
                        fontweight='bold'
                    )

                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")

                    plt.tight_layout()

                    st.pyplot(
                        fig3,
                        use_container_width=True
                    )

                    plt.close(fig3)


        # ========================================================
        # KDE
        # ========================================================

        st.html(
            """
            <div class="distribution-section">

                <div class="distribution-title">
                    📈 Kernel Density Estimation
                </div>

                <div class="distribution-description">
                    KDE provides a smoothed estimate of the probability
                    density of numerical features.
                </div>

            </div>
            """
        )

        for start in range(0, len(numerical_features), 2):

            current_features = numerical_features[start:start + 2]

            plot_columns = st.columns(2)

            for column, feature in zip(plot_columns, current_features):

                with column:

                    st.html(
                        f"""
                        <div class="distribution-card">

                            <div class="distribution-card-title">
                                📉 {feature}
                            </div>

                            <div class="distribution-card-subtitle">
                                Kernel Density Estimate
                            </div>

                        </div>
                        """
                    )

                    fig4, ax = plt.subplots(figsize=(6, 4))

                    sns.kdeplot(
                        uploaded_data[feature],
                        shade=True,
                        ax=ax
                    )

                    ax.set_title(
                        f"{feature} Density",
                        fontsize=12,
                        fontweight='bold'
                    )

                    ax.set_xlabel(feature)
                    ax.set_ylabel("Density")

                    plt.tight_layout()

                    st.pyplot(
                        fig4,
                        use_container_width=True
                    )

                    plt.close(fig4)


        # ========================================================
        # DISTPLOT
        # ========================================================

        st.html(
            """
            <div class="distribution-section">

                <div class="distribution-title">
                    📊 Distribution Plot
                </div>

                <div class="distribution-description">
                    Combined view of the numerical feature distributions.
                </div>

            </div>
            """
        )

        for start in range(0, len(numerical_features), 2):

            current_features = numerical_features[start:start + 2]

            plot_columns = st.columns(2)

            for column, feature in zip(plot_columns, current_features):

                with column:

                    st.html(
                        f"""
                        <div class="distribution-card">

                            <div class="distribution-card-title">
                                📊 {feature}
                            </div>

                            <div class="distribution-card-subtitle">
                                Distribution Plot
                            </div>

                        </div>
                        """
                    )

                    fig4, ax = plt.subplots(figsize=(6, 4))

                    sns.distplot(
                        uploaded_data[feature],
                        ax=ax,
                        bins=5
                    )

                    ax.set_title(
                        f"{feature} Distribution Plot",
                        fontsize=12,
                        fontweight='bold'
                    )

                    ax.set_xlabel(feature)
                    ax.set_ylabel("Density")

                    plt.tight_layout()

                    st.pyplot(
                        fig4,
                        use_container_width=True
                    )

                    plt.close(fig4)
        
    
    if selected == "Outliers Detection":

        # ========================================================
        # OUTLIERS DETECTION
        # ========================================================

        st.html(
            """
            <div class="outlier-section">

                <div class="outlier-title">
                    🔎 IQR Outlier Detection
                </div>

                <div class="outlier-description">
                    Identify potential outliers in numerical features using
                    the Interquartile Range (IQR) method.
                </div>

            </div>
            """
        )


        # ========================================================
        # IQR
        # ========================================================

        for feature in numerical_features:

            Q1 = uploaded_data[feature].quantile(0.25)
            Q3 = uploaded_data[feature].quantile(0.75)

            IQR = Q3 - Q1

            lower_boundry = Q1 - 1.5 * IQR
            upper_boundry = Q3 + 1.5 * IQR

            outliers = uploaded_data[
                (uploaded_data[feature] < lower_boundry) |
                (uploaded_data[feature] > upper_boundry)
            ]

            index_of_outliers = outliers.index


            # ====================================================
            # FEATURE CARD
            # ====================================================

            st.html(
                f"""
                <div class="outlier-card">

                    <div class="outlier-card-title">
                        📊 {feature}
                    </div>

                    <div class="outlier-count 
                        {'outlier-count-danger' if outliers.shape[0] > 0 else 'outlier-count-success'}">

                        {outliers.shape[0]} outlier(s) detected

                    </div>

                    <div class="outlier-boundary">
                        Lower boundary:
                        <strong>{lower_boundry:.2f}</strong>

                        &nbsp;&nbsp; | &nbsp;&nbsp;

                        Upper boundary:
                        <strong>{upper_boundry:.2f}</strong>
                    </div>

                    <div class="outlier-values-title">
                        Values of the Outliers in the Dataset
                    </div>

                </div>
                """
            )


            # ====================================================
            # OUTLIER VALUES
            # ====================================================

            if outliers.shape[0] > 0:

                outlier_values = pd.DataFrame({
                    feature: outliers[feature]
                })

                st.dataframe(
                    outlier_values,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.html(
                    """
                    <div class="outlier-empty">
                        ✓ No outliers were detected for this feature.
                    </div>
                    """
                )

            st.write("")

        

    if selected == "Feature Analysis":

        # ========================================================
        # FEATURE ANALYSIS
        # ========================================================

        st.html(
            """
            <div class="feature-section">

                <div class="feature-section-title">
                    🔬 Feature Analysis
                </div>

                <div class="feature-section-description">
                    Explore the distribution, spread, and relationships
                    between numerical features.
                </div>

            </div>
            """
        )


        # ========================================================
        # BOX PLOT
        # ========================================================

        st.html(
            """
            <div class="feature-section">

                <div class="feature-section-title">
                    📦 Box Plot
                </div>

                <div class="feature-section-description">
                    Visualize the spread, median, quartiles, and potential
                    outliers of each numerical feature.
                </div>

            </div>
            """
        )

        for start in range(0, len(numerical_features), 2):

            current_features = numerical_features[start:start + 2]

            plot_columns = st.columns(2)

            for column, feature in zip(plot_columns, current_features):

                with column:

                    st.html(
                        f"""
                        <div class="feature-card">

                            <div class="feature-card-title">
                                📦 {feature}
                            </div>

                            <div class="feature-card-subtitle">
                                Box Plot
                            </div>

                        </div>
                        """
                    )

                    fig5, ax = plt.subplots(figsize=(5.5, 4))

                    sns.boxplot(
                        y=uploaded_data[feature],
                        ax=ax
                    )

                    ax.set_ylabel(feature)

                    plt.tight_layout()

                    st.pyplot(
                        fig5,
                        use_container_width=True
                    )

                    plt.close(fig5)


        # ========================================================
        # VIOLIN PLOT
        # ========================================================

        st.html(
            """
            <div class="feature-section">

                <div class="feature-section-title">
                    🎻 Violin Plot
                </div>

                <div class="feature-section-description">
                    Examine the distribution and density of each numerical
                    feature.
                </div>

            </div>
            """
        )

        for start in range(0, len(numerical_features), 2):

            current_features = numerical_features[start:start + 2]

            plot_columns = st.columns(2)

            for column, feature in zip(plot_columns, current_features):

                with column:

                    st.html(
                        f"""
                        <div class="feature-card">

                            <div class="feature-card-title">
                                🎻 {feature}
                            </div>

                            <div class="feature-card-subtitle">
                                Violin Plot
                            </div>

                        </div>
                        """
                    )

                    fig6, ax = plt.subplots(figsize=(5.5, 4))

                    sns.violinplot(
                        uploaded_data[feature],
                        ax=ax
                    )

                    ax.set_ylabel(feature)

                    plt.tight_layout()

                    st.pyplot(
                        fig6,
                        use_container_width=True
                    )

                    plt.close(fig6)


        # ========================================================
        # SCATTER PLOT
        # ========================================================

        st.html(
            """
            <div class="feature-section">

                <div class="feature-section-title">
                    🔵 Scatter Plot
                </div>

                <div class="feature-section-description">
                    Explore the relationship between two numerical features.
                </div>

            </div>
            """
        )


        # ========================================================
        # FEATURE SELECTION
        # ========================================================

        selection_columns = st.columns(2)

        with selection_columns[0]:

            st.html(
                """
                <div class="scatter-card">

                    <div class="scatter-card-title">
                        X-Axis Feature
                    </div>

                    <div class="scatter-card-description">
                        Select the feature for the horizontal axis.
                    </div>

                </div>
                """
            )

            feature1 = st.selectbox(
                'Select feature X',
                options=numerical_features
            )


        with selection_columns[1]:

            st.html(
                """
                <div class="scatter-card">

                    <div class="scatter-card-title">
                        Y-Axis Feature
                    </div>

                    <div class="scatter-card-description">
                        Select the feature for the vertical axis.
                    </div>

                </div>
                """
            )

            feature2 = st.selectbox(
                'Select Feature Y',
                options=numerical_features
            )


        # ========================================================
        # SCATTER DATA
        # ========================================================

        feature1_data = uploaded_data[feature1]
        feature2_data = uploaded_data[feature2]


        # ========================================================
        # SCATTER PLOT
        # ========================================================

        st.html(
            f"""
            <div class="scatter-card">

                <div class="scatter-card-title">
                    🔵 {feature1} vs {feature2}
                </div>

                <div class="scatter-card-description">
                    Relationship between the selected numerical features.
                </div>

            </div>
            """
        )

        fig7, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            x=feature1_data,
            y=feature2_data,
            ax=ax
        )

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)

        plt.tight_layout()

        st.pyplot(
            fig7,
            use_container_width=True
        )

        plt.close(fig7)
    

    if selected == "LLM":

        # ========================================================
        # LLM ANALYSIS
        # ========================================================

        st.html(
            """
            <div class="llm-section">

                <div class="llm-title">
                    🤖 AI-Powered Dataset Analysis
                </div>

                <div class="llm-description">
                    Use artificial intelligence to understand your dataset,
                    discover important patterns, and receive machine learning
                    recommendations.
                </div>

            </div>
            """
        )


        # ========================================================
        # CREATE THREE COLUMNS
        # ========================================================

        llm_columns = st.columns(3)


        # ========================================================
        # DATASET SUMMARY
        # ========================================================

        with llm_columns[0]:

            st.html(
                """
                <div class="llm-card llm-card-primary">

                    <div class="llm-icon">
                        📊
                    </div>

                    <div class="llm-card-title">
                        Dataset Summary
                    </div>

                    <div class="llm-card-description">
                        Get an AI-generated overview of your dataset,
                        including important statistical characteristics
                        and potential patterns.
                    </div>

                </div>
                """
            )

            if st.button(
                "Generate Dataset Summary",
                key="llm_dataset_summary"
            ):

                with st.spinner("Analyzing your dataset..."):

                    output_of_LLM_phase3_part1 = eda_scan(eda_results)

                st.html(
                    """
                    <div class="llm-result">

                        <div class="llm-result-title">
                            🤖 AI Dataset Analysis
                        </div>

                    </div>
                    """
                )

                st.markdown(output_of_LLM_phase3_part1)


        # ========================================================
        # FEATURE INSIGHTS
        # ========================================================

        with llm_columns[1]:

            st.html(
                """
                <div class="llm-card llm-card-secondary">

                    <div class="llm-icon">
                        🔍
                    </div>

                    <div class="llm-card-title">
                        Feature Insights
                    </div>

                    <div class="llm-card-description">
                        Let the AI examine individual features and
                        identify meaningful characteristics, patterns,
                        and potential issues.
                    </div>

                </div>
                """
            )

            if st.button(
                "Generate Feature Insights",
                key="llm_feature_insights"
            ):

                with st.spinner("Analyzing your features..."):

                    output_of_LLM_phase3_part2 = eda_scan_each_feature(
                        each_feature_summary
                    )

                st.html(
                    """
                    <div class="llm-result">

                        <div class="llm-result-title">
                            🔍 AI Feature Analysis
                        </div>

                    </div>
                    """
                )

                st.markdown(output_of_LLM_phase3_part2)


        # ========================================================
        # MACHINE LEARNING MODEL SUGGESTION
        # ========================================================

        with llm_columns[2]:

            st.html(
                """
                <div class="llm-card llm-card-tertiary">

                    <div class="llm-icon">
                        🧠
                    </div>

                    <div class="llm-card-title">
                        ML Model Suggestion
                    </div>

                    <div class="llm-card-description">
                        Receive AI-powered recommendations for suitable
                        machine learning approaches based on your dataset.
                    </div>

                </div>
                """
            )

            if st.button(
                "Machine Learning Model Suggestion",
                key="llm_model_suggestion"
            ):

                with st.spinner("Finding suitable models..."):

                    output_of_LLM_phase3_part3 = model_recommendation(
                        each_feature_summary,
                        eda_results
                    )

                st.html(
                    """
                    <div class="llm-result">

                        <div class="llm-result-title">
                            🧠 AI Model Recommendation
                        </div>

                    </div>
                    """
                )

                st.markdown(output_of_LLM_phase3_part3)

    
    if selected == "Chat with AI":

        # ========================================================
        # CHAT WITH AI
        # ========================================================

        st.html(
            """
            <div class="chat-section">

                <div class="chat-title">
                    💬 Chat with AI
                </div>

                <div class="chat-description">
                    Ask questions about your dataset, its features,
                    statistical patterns, or machine learning possibilities.
                </div>

            </div>
            """
        )


        # ========================================================
        # INFORMATION CARD
        # ========================================================

        st.html(
            """
            <div class="chat-info-card">

                <div class="chat-info-icon">
                    🤖
                </div>

                <div class="chat-info-title">
                    Your AI Data Science Assistant
                </div>

                <div class="chat-info-text">
                    Ask me anything about your dataset. I can help you
                    understand statistical results, identify important
                    patterns, and answer questions based on your analysis.
                </div>

            </div>
            """
        )


        # ========================================================
        # SESSION STATE
        # ========================================================

        question = st.chat_input(
            "Ask something about your dataset..."
        )


        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []


        if "eda_results" not in st.session_state:
            st.session_state.eda_results = []


        if "each_feature_summary" not in st.session_state:
            st.session_state.each_feature_summary = []


        st.session_state.eda_results = eda_results
        st.session_state.each_feature_summary = each_feature_summary


        # ========================================================
        # CHAT HISTORY
        # ========================================================

        if len(st.session_state.chat_history) == 0 and not question:

            st.html(
                """
                <div class="chat-empty">

                    <div class="chat-empty-icon">
                        💡
                    </div>

                    <div class="chat-empty-title">
                        Start a conversation
                    </div>

                    <div class="chat-empty-text">
                        Try asking: "What are the most important features
                        in my dataset?"
                    </div>

                </div>
                """
            )


        # ========================================================
        # NEW QUESTION
        # ========================================================

        if question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            # ====================================================
            # AI RESPONSE
            # ====================================================

            with st.spinner("AI is analyzing your question..."):

                answer = ai_answer(
                    question,
                    st.session_state.eda_results,
                    st.session_state.each_feature_summary,
                    st.session_state.chat_history
                )


            st.session_state.chat_history.append(
                {
                    "role": "AI",
                    "content": answer
                }
            )


        # ========================================================
        # DISPLAY CHAT HISTORY
        # ========================================================

        for index, message in enumerate(
            st.session_state.chat_history,
            start=1
        ):

            with st.chat_message(message["role"]):

                st.markdown(message["content"])



    if selected == "AI Assistant":

        # ========================================================
        # AI ASSISTANT / MACHINE LEARNING
        # ========================================================

        st.html(
            """
            <div class="ml-section">

                <div class="ml-title">
                    🧠 AI Assistant
                </div>

                <div class="ml-description">
                    Select a target feature and problem type to train and
                    compare multiple machine learning models.
                </div>

            </div>
            """
        )


        # ========================================================
        # MODEL CONFIGURATION
        # ========================================================

        st.html(
            """
            <div class="ml-control-card">

                <div class="ml-control-title">
                    ⚙️ Model Configuration
                </div>

                <div class="ml-control-description">
                    Choose the feature you want to predict and the type
                    of machine learning problem.
                </div>

            </div>
            """
        )


        target_feature = st.selectbox(
            '**select which feature is going to be your target**',
            (uploaded_data.columns)
        )


        problem_type = st.radio(
            "**select problem type**",
            ('Classification', 'Regression'),
            horizontal=True
        )


        st.html(
            f"""
            <div class="ml-target-badge">
                🎯 Target: {target_feature}
            </div>
            """
        )


        # ========================================================
        # CLASSIFICATION
        # ========================================================

        if problem_type == 'Classification':

            set_session_state(target_feature, problem_type)

            st.html(
                """
                <div class="ml-run-card">

                    <div class="ml-run-title">
                        🚀 Classification Models
                    </div>

                    <div class="ml-run-description">
                        Compare Logistic Regression, Random Forest,
                        SVC, and KNN classifiers.
                    </div>

                </div>
                """
            )


            button_run_machine_learning_algorithms = st.button(
                "Run Machine Learning",
                key="run_classification_models"
            )


            uploaded_data = uploaded_data.dropna(
                subset=[target_feature]
            )


            train_test_splited = preprocessing_before_training_models(
                target_feature,
                numerical_features,
                categorical_features,
                uploaded_data
            )


            # ====================================================
            # Logistic Regression
            # ====================================================

            evaluation_metrics_for_logistic_regression = 0

            if problem_type == "Classification" and pd.api.types.is_float_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Warning: Logistic Regression cannot have float labels, "
                    "It is better to choose another feature as the target feature"
                )

            else:

                logistic_reg_y_pred = logistic_regression(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                conf_matrix = confusion_matrixx(
                    train_test_splited[3],
                    logistic_reg_y_pred
                )

                cls_report = classification_rep(
                    train_test_splited[3],
                    logistic_reg_y_pred
                )

                evaluation_metrics_for_logistic_regression = metrics(
                    logistic_reg_y_pred,
                    train_test_splited[3]
                )


            # ====================================================
            # Random Forest Classifier
            # ====================================================

            evaluation_metrics_for_Randon_Forest = 0

            if problem_type == 'Classification' and pd.api.types.is_float_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Warning: Random Forest Classifier cannot have float labels, "
                    "It is better to choose another feature as the target feature"
                )

            else:

                random_forest_classifier_y_pred = random_forest_calssifer(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                conf_matrix = confusion_matrixx(
                    train_test_splited[3],
                    random_forest_classifier_y_pred
                )

                cls_report = classification_rep(
                    train_test_splited[3],
                    random_forest_classifier_y_pred
                )

                evaluation_metrics_for_Randon_Forest = metrics(
                    random_forest_classifier_y_pred,
                    train_test_splited[3]
                )


            # ====================================================
            # SVC Classifier
            # ====================================================

            evaluation_metrics_for_SVC = 0

            if problem_type == 'Classification' and pd.api.types.is_float_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Warning: SVC Classifier cannot have float labels, "
                    "It is better to choose another feature as the target feature"
                )

            else:

                svc_classifier_y_pred = svc_classifier(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                conf_matrix = confusion_matrixx(
                    train_test_splited[3],
                    svc_classifier_y_pred
                )

                cls_report = classification_rep(
                    train_test_splited[3],
                    svc_classifier_y_pred
                )

                evaluation_metrics_for_SVC = metrics(
                    svc_classifier_y_pred,
                    train_test_splited[3]
                )


            # ====================================================
            # KNN Classifier
            # ====================================================

            evaluation_metrics_for_KNN = 0

            if problem_type == 'Classification' and pd.api.types.is_float_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Warning: KNN Classifier cannot have float labels, "
                    "It is better to choose another feature as the target feature"
                )

            else:

                KNN_classifier_y_pred = knn_classifier(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                conf_matrix = confusion_matrixx(
                    train_test_splited[3],
                    KNN_classifier_y_pred
                )

                cls_report = classification_rep(
                    train_test_splited[3],
                    KNN_classifier_y_pred
                )

                evaluation_metrics_for_KNN = metrics(
                    KNN_classifier_y_pred,
                    train_test_splited[3]
                )


            # ====================================================
            # CLASSIFICATION RESULTS
            # ====================================================

            if button_run_machine_learning_algorithms:

                st.html(
                    """
                    <div class="ml-results-header">
                        📊 Classification Model Comparison
                    </div>

                    <div class="ml-results-description">
                        Compare the performance of the selected
                        classification algorithms.
                    </div>
                    """
                )


                conf_matrix = pd.DataFrame(
                    {
                        "Precision": [
                            evaluation_metrics_for_logistic_regression["precision"],
                            evaluation_metrics_for_Randon_Forest["precision"],
                            evaluation_metrics_for_SVC["precision"],
                            evaluation_metrics_for_KNN["precision"]
                        ],

                        "Recall": [
                            evaluation_metrics_for_logistic_regression["recall"],
                            evaluation_metrics_for_Randon_Forest["recall"],
                            evaluation_metrics_for_SVC["recall"],
                            evaluation_metrics_for_KNN["recall"]
                        ],

                        "F1-score": [
                            evaluation_metrics_for_logistic_regression["f1_scoree"],
                            evaluation_metrics_for_Randon_Forest["f1_scoree"],
                            evaluation_metrics_for_SVC["f1_scoree"],
                            evaluation_metrics_for_KNN["f1_scoree"]
                        ],

                        "Accuracy": [
                            evaluation_metrics_for_logistic_regression["accuracy"],
                            evaluation_metrics_for_Randon_Forest["accuracy"],
                            evaluation_metrics_for_SVC["accuracy"],
                            evaluation_metrics_for_KNN["accuracy"]
                        ],
                    },

                    index=[
                        "Logistic_Regression",
                        "Random Forest",
                        "SVC",
                        "KNN"
                    ],
                )


                st.html(
                    '<div class="ml-results-table">'
                )

                st.dataframe(
                    conf_matrix,
                    use_container_width=True
                )

                st.html("</div>")


        # ========================================================
        # REGRESSION
        # ========================================================

        if problem_type == 'Regression':

            set_session_state(target_feature, problem_type)

            st.html(
                """
                <div class="ml-run-card">

                    <div class="ml-run-title">
                        📈 Regression Models
                    </div>

                    <div class="ml-run-description">
                        Compare Linear Regression, Ridge Regression,
                        Lasso Regression, and Huber Regression.
                    </div>

                </div>
                """
            )


            button_run_machine_learning_algorithms = st.button(
                "Run Machine Learning",
                key="run_regression_models"
            )


            uploaded_data = uploaded_data.dropna(
                subset=[target_feature]
            )


            train_test_splited = preprocessing_before_training_models(
                target_feature,
                numerical_features,
                categorical_features,
                uploaded_data
            )


            # ====================================================
            # Linear Regression
            # ====================================================

            evaluation_report_linear_models_linear_reg = 0

            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Linear Regression Model cannot handle target features "
                    "with categorical values"
                )

            else:

                linear_reg_y_pred = linear_models(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                evaluation_report_linear_models_linear_reg = report_for_linear_models(
                    train_test_splited[3],
                    linear_reg_y_pred["linear_regression"]
                )


            # ====================================================
            # Ridge Regression
            # ====================================================

            evaluation_report_linear_models_ridge = 0

            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Ridge Regression Model cannot handle target features "
                    "with categorical values"
                )

            else:

                ridge_regressor_y_pred = linear_models(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                evaluation_report_linear_models_ridge = report_for_linear_models(
                    train_test_splited[3],
                    ridge_regressor_y_pred["ridge"]
                )


            # ====================================================
            # Lasso Regression
            # ====================================================

            evaluation_report_linear_models_lasso = 0

            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Lasso Regression Model cannot handle target features "
                    "with categorical values"
                )

            else:

                lasso_regressor_y_pred = linear_models(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                evaluation_report_linear_models_lasso = report_for_linear_models(
                    train_test_splited[3],
                    lasso_regressor_y_pred["lasso"]
                )


            # ====================================================
            # Huber Regressor
            # ====================================================

            evaluation_report_linear_models_huber = 0

            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(
                uploaded_data[target_feature]
            ):

                st.warning(
                    "Huber Regression Model cannot handle target features "
                    "with categorical values"
                )

            else:

                huber_regressor_y_pred = linear_models(
                    train_test_splited[0],
                    train_test_splited[1],
                    train_test_splited[2]
                )

                evaluation_report_linear_models_huber = report_for_linear_models(
                    train_test_splited[3],
                    huber_regressor_y_pred["huber_regressor"]
                )


            # ====================================================
            # REGRESSION RESULTS
            # ====================================================

            if button_run_machine_learning_algorithms:

                st.html(
                    """
                    <div class="ml-results-header">
                        📊 Regression Model Comparison
                    </div>

                    <div class="ml-results-description">
                        Compare the performance of the selected
                        regression algorithms using standard evaluation metrics.
                    </div>
                    """
                )


                conf_matrix = pd.DataFrame(
                    {
                        "MSE": [
                            evaluation_report_linear_models_linear_reg["mse"],
                            evaluation_report_linear_models_ridge["mse"],
                            evaluation_report_linear_models_huber["mse"],
                            evaluation_report_linear_models_lasso['mse']
                        ],

                        "Rmse": [
                            evaluation_report_linear_models_linear_reg["rmse"],
                            evaluation_report_linear_models_ridge["rmse"],
                            evaluation_report_linear_models_huber["rmse"],
                            evaluation_report_linear_models_lasso['rmse']
                        ],

                        "R2": [
                            evaluation_report_linear_models_linear_reg["r2"],
                            evaluation_report_linear_models_ridge["r2"],
                            evaluation_report_linear_models_huber["r2"],
                            evaluation_report_linear_models_lasso['r2']
                        ],

                        "MAE": [
                            evaluation_report_linear_models_linear_reg["mae"],
                            evaluation_report_linear_models_ridge['mae'],
                            evaluation_report_linear_models_huber['mae'],
                            evaluation_report_linear_models_lasso['mae']
                        ]

                    },

                    index=[
                        "Linear Regression",
                        "Ridge Regression",
                        "Huber Regressor",
                        "Lasso Regression"
                    ],
                )


                st.html(
                    '<div class="ml-results-table">'
                )

                st.dataframe(
                    conf_matrix,
                    use_container_width=True
                )

                st.html("</div>")

    


        

    







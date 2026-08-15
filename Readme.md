# 📊 Copilot Data Scientist Assistant

An AI-powered Data Science assistant built with **Python and Streamlit**.

The application helps users explore datasets, perform EDA, get AI-generated insights, chat with an AI assistant, and train and compare Machine Learning models.

---

## 🚀 Features

### 📂 Dataset Upload

Upload datasets in:

* CSV

The application automatically analyzes:

* Dataset size
* Data types
* Numerical and categorical features
* Missing values
* Descriptive statistics

### 📊 Exploratory Data Analysis

The application provides:

* Dataset statistics
* Missing-value analysis
* Pearson correlation
* Spearman correlation
* Histograms
* KDE plots
* Distribution plots
* Box plots
* Violin plots
* Scatter plots
* Outlier detection

### 🚨 Outlier Detection

Potential outliers are detected using the **Interquartile Range (IQR)** method.

IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

Values outside these boundaries are identified as potential outliers.

### 🤖 LLM Analysis

The application uses a Large Language Model to analyze information extracted from the EDA process.

The LLM can provide:

* Dataset summaries
* Feature insights
* Machine Learning recommendations

Instead of sending the entire dataset to the LLM, the application sends relevant EDA information in structured dictionaries.

### 💬 AI Chat

Users can ask questions about their dataset and receive AI-generated answers based on the available EDA information.

Conversation history is maintained using Streamlit session state.

### 🧠 Machine Learning

Users can manually select:

* Target feature
* Classification or Regression

The application then preprocesses the data and trains multiple models.

#### Classification

* Logistic Regression
* Random Forest Classifier
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score

#### Regression

* Linear Regression
* Ridge Regression
* Lasso Regression
* Huber Regressor

Evaluation metrics:

* MSE
* RMSE
* R²
* MAE

---

## 🔄 Workflow

```text
Upload Dataset
      ↓
EDA
      ↓
Statistics & Visualizations
      ↓
Outlier Detection
      ↓
LLM Analysis
      ↓
AI Chat
      ↓
Machine Learning
      ↓
Model Comparison
```

---

## 🛠️ Technologies

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Large Language Models (LLMs)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/copilot-data-scientist-assistant.git
cd copilot-data-scientist-assistant
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key

The LLM functionality requires an API key.

For example:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Do not upload your API key to GitHub.

If you use a `.env` file, make sure it is included in `.gitignore`.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🎯 Goal

The goal of this project is to combine:

```text
Data Science
    +
Machine Learning
    +
Large Language Models
    +
Interactive Visualization
```

into a single **Data Scientist Copilot**.

The application is designed to assist Data Scientists with exploration, analysis, interpretation, and Machine Learning experimentation.

---

## 👨‍💻 Author

**Amirali Ghanbarpourshiadeh**

Computer Engineering → Data Science & Artificial Intelligence

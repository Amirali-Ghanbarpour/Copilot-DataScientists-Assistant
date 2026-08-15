# 📊 Copilot Data Scientist Assistant

An interactive AI-powered Data Science assistant built with **Python and Streamlit** that helps users explore datasets, perform Exploratory Data Analysis (EDA), obtain AI-generated insights, interact with an AI assistant, and train and compare multiple Machine Learning models.

The project is designed to bring several common Data Science workflows together into a single user-friendly interface.

---

## 🚀 Overview

The **Copilot Data Scientist Assistant** is an end-to-end Data Science application that assists users throughout several stages of the data analysis workflow.

Instead of manually performing every step of dataset exploration, statistical analysis, visualization, and model evaluation, users can upload their dataset and interact with different modules of the application.

The application currently provides:

- Dataset exploration and statistical analysis
- Correlation analysis
- Distribution analysis
- Outlier detection
- Feature analysis and visualization
- LLM-powered dataset insights
- AI-powered conversational analysis
- Automated Machine Learning model training and comparison

The project combines traditional Data Science techniques with Large Language Models (LLMs) to create an interactive **Data Scientist Copilot**.

---

1. 📂 Dataset Upload

Users can upload their datasets directly through the Streamlit interface.

Supported formats:

CSV

Excel (.xlsx)

After uploading a dataset, the application automatically identifies:

Numerical features

Categorical features

Dataset dimensions

Missing values

Statistical information

2. 📈 Dataset Statistics

The application provides an overview of the uploaded dataset.

The following information is displayed:

Number of rows

Number of columns

Column data types

Missing values

Mean

Minimum

Maximum

Descriptive statistics

Numerical features are presented through visually organized statistic cards.

3. 🔗 Correlation Analysis

The Correlation Analysis module provides two different statistical methods for analyzing relationships between numerical features.

Pearson Correlation

Measures the linear relationship between numerical variables.

Spearman Correlation

Measures the monotonic relationship between numerical variables.

For both methods, the application provides:

Correlation matrices

Correlation heatmaps

The module also provides missing-value analysis, including:

Missing values by feature

Missing-value percentages

Missing-value visualization

Dataset descriptive statistics

4. 📊 Distribution Analysis

The Distribution Analysis module provides several visualization techniques for understanding the distribution of dataset features.

Histogram

Used to visualize the frequency distribution of values within a feature.

Kernel Density Estimation (KDE)

Used to estimate and visualize the probability density of numerical features.

Distribution Plot

Provides another visualization of the distribution and spread of numerical variables.

These visualizations help users understand:

Data distribution

Spread

Skewness

Potential anomalies

General patterns within the data

5. 🚨 Outlier Detection

The application detects potential outliers using the Interquartile Range (IQR) method.

For each numerical feature, the following calculations are performed:

IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

Values below the Lower Bound or above the Upper Bound are identified as potential outliers.

For each numerical feature, the application displays:

Number of detected outliers

Values of the detected outliers

This allows users to quickly identify unusual observations within their dataset.

6. 🔬 Feature Analysis

The Feature Analysis module provides several visual tools for investigating numerical variables.

Box Plots

Useful for analyzing:

Distribution

Median

Quartiles

Potential outliers

Violin Plots

Useful for understanding:

Shape of the distribution

Data density

Spread of numerical features

Scatter Plots

Users can select two numerical features and visualize their relationship.

This allows users to investigate potential:

Correlations

Trends

Clusters

Non-linear relationships

7. 🤖 LLM-Powered Data Analysis

The application integrates a Large Language Model into the Data Science workflow.

Instead of directly sending the entire dataset to the LLM, the application extracts relevant information during the Exploratory Data Analysis stage and organizes it into structured dictionaries.

The information provided to the LLM includes:

Statistical summaries

Missing values

Correlations

Outliers

Feature-level statistics

Feature relationships

This approach allows the LLM to focus on relevant analytical information while avoiding unnecessary transfer of the complete dataset.

Dataset Summary

The LLM analyzes the collected EDA information and generates a natural-language summary of the dataset.

Feature Insights

The LLM analyzes individual features and provides interpretations and insights.

Machine Learning Model Recommendation

The LLM analyzes the characteristics of the dataset and provides suggestions regarding suitable Machine Learning approaches.

8. 💬 Chat with AI

The application provides an interactive conversational AI interface.

Users can ask questions about their dataset and receive AI-generated answers based on the previously calculated EDA information.

The AI receives:

User questions

EDA results

Feature summaries

Previous conversation history

The application maintains the conversation history using Streamlit session state.

This allows users to interact with their dataset using natural language instead of relying only on static visualizations.

9. 🧠 AI Assistant & Machine Learning

The AI Assistant module allows users to select a target feature and choose the type of Machine Learning problem.

The available problem types are:

Classification

Regression

After selecting the target feature and problem type, the application preprocesses the dataset and trains multiple Machine Learning models.

The results of the different models are then compared using appropriate evaluation metrics.

Classification

The application currently implements the following classification algorithms:

Logistic Regression

Random Forest Classifier

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Classification Evaluation

The classification models are compared using:

Precision

Recall

F1-score

Accuracy

The results are presented in a comparison table to help users evaluate the performance of the different classification algorithms.

Regression

The application currently implements the following regression algorithms:

Linear Regression

Ridge Regression

Lasso Regression

Huber Regressor

Regression Evaluation

The regression models are evaluated using:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Mean Absolute Error (MAE)

The results are presented in a comparison table to allow users to compare the performance of the different regression models.

10. 🔄 Data Processing Workflow

The overall workflow of the application is:

Dataset Upload

↓

Feature Type Detection

↓

Exploratory Data Analysis

↓

Dataset Statistics

↓

Correlation Analysis

↓

Distribution Analysis

↓

Outlier Detection

↓

Feature Analysis

↓

LLM Analysis

↓

AI Chat

↓

Machine Learning

↓

Model Evaluation

This workflow combines traditional Data Science techniques with Artificial Intelligence and Machine Learning in a single interactive application.


# 6. 🔬 Feature Analysis

The Feature Analysis module provides several visual tools for investigating numerical variables.

### Box Plots

Useful for analyzing:

* Distribution
* Median
* Quartiles
* Potential outliers

### Violin Plots

Useful for understanding the shape and density of feature distributions.

### Scatter Plots

Users can select two numerical features and visualize their relationship.

This allows users to investigate potential:

* Correlations
* Trends
* Clusters
* Non-linear relationships

---

# 7. 🤖 LLM-Powered Data Analysis

One of the main components of the project is the integration of a Large Language Model into the Data Science workflow.

The application does not simply send the entire dataset to the LLM.

Instead, relevant information extracted during the EDA process is organized into structured dictionaries and provided to the AI.

This includes information such as:

* Statistical summaries
* Missing values
* Correlations
* Outliers
* Feature-level statistics
* Feature relationships

This approach reduces unnecessary data transfer and allows the LLM to focus on the information that is relevant for analysis.

---

## LLM Capabilities

### 📋 Dataset Summary

The LLM analyzes the collected EDA information and generates a natural-language summary of the dataset.

### 🔍 Feature Insights

The LLM analyzes individual features and provides interpretations and insights.

### 🧠 Machine Learning Model Recommendation

The LLM analyzes the available dataset characteristics and provides suggestions regarding suitable Machine Learning approaches.

---

# 8. 💬 Chat with AI

The application includes an interactive conversational AI interface.

Users can ask questions about their dataset and receive AI-generated answers based on the previously calculated EDA information.

The application maintains conversation history using Streamlit session state.

The AI receives:

* User questions
* EDA results
* Feature summaries
* Previous conversation history

This allows users to interact with the dataset conversationally rather than relying only on static visualizations.

---

# 9. 🧠 AI Assistant & Machine Learning

The AI Assistant module allows users to select a target feature and choose between:

* Classification
* Regression

The application then preprocesses the dataset and trains multiple Machine Learning models.

---

## Classification

The following classification algorithms are implemented:

### Logistic Regression

A linear classification algorithm used as a baseline model.

### Random Forest Classifier

An ensemble learning method based on multiple decision trees.

### Support Vector Classifier (SVC)

A supervised learning algorithm that finds decision boundaries between classes.

### K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts classes based on neighboring observations.

---

## Classification Evaluation

The models are compared using:

* Precision
* Recall
* F1-score
* Accuracy

The results are presented in a comparison table so that users can easily evaluate the performance of the different algorithms.

---

# 10. 📈 Regression

For regression problems, the application implements:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Huber Regressor

These models provide different approaches for predicting continuous numerical targets.

---

## Regression Evaluation

The regression models are evaluated using:

### Mean Squared Error (MSE)

Measures the average squared difference between predictions and actual values.

### Root Mean Squared Error (RMSE)

The square root of MSE, expressed in the same units as the target variable.

### R² Score

Measures how much of the variance in the target variable is explained by the model.

### Mean Absolute Error (MAE)

Measures the average absolute difference between predictions and actual values.

The models are presented in a comparison table to facilitate model selection.

---

# 🔄 Data Processing Pipeline

The general workflow of the application can be summarized as:

```text
             Dataset Upload
                    │
                    ▼
          Feature Type Detection
                    │
                    ▼
        ┌─────────────────────────┐
        │   Exploratory Data      │
        │       Analysis          │
        └─────────────────────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Statistics   Correlation   Distributions
       │            │            │
       └────────────┼────────────┘
                    ▼
            Outlier Detection
                    │
                    ▼
             Feature Analysis
                    │
                    ▼
          ┌───────────────────┐
          │   LLM Analysis    │
          └───────────────────┘
             │             │
             ▼             ▼
       AI Insights     AI Chatbot
                           │
                           ▼
                  Machine Learning
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
             Classification      Regression
                  │                 │
                  ▼                 ▼
             Model Metrics      Model Metrics
```

---

# 🏗️ Project Architecture

The project is organized around several major components:

```text
Copilot Data Scientist Assistant
│
├── Dataset Input
│
├── Exploratory Data Analysis
│   ├── Dataset Statistics
│   ├── Missing Values
│   ├── Correlation Analysis
│   ├── Distribution Analysis
│   ├── Outlier Detection
│   └── Feature Analysis
│
├── LLM Layer
│   ├── Dataset Summary
│   ├── Feature Insights
│   └── Model Recommendations
│
├── Conversational AI
│   └── Dataset-aware Chatbot
│
└── Machine Learning
    ├── Preprocessing
    ├── Classification
    │   ├── Logistic Regression
    │   ├── Random Forest
    │   ├── SVC
    │   └── KNN
    │
    └── Regression
        ├── Linear Regression
        ├── Ridge Regression
        ├── Lasso Regression
        └── Huber Regressor
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## User Interface

* Streamlit
* Streamlit Option Menu
* Custom CSS

## Data Processing

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn

## Artificial Intelligence

* Large Language Models (LLMs)
* LLM API integration
* AI-powered dataset analysis
* Conversational AI

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/copilot-data-scientist-assistant.git
```

Move into the project directory:

```bash
cd copilot-data-scientist-assistant
```

---

## 2. Create a virtual environment

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

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 API Configuration

The LLM functionality requires an API key.

Create an environment variable or configure the API key according to the implementation in the project.

For example:

```text
OPENROUTER_API_KEY=your_api_key_here
```

**Do not commit API keys or other secrets to GitHub.**

A `.env` file can be used locally:

```text
.env
```

and should be included in `.gitignore`.

---

# ▶️ Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

Then open the URL displayed in the terminal.

Usually, Streamlit runs the application at:

```text
http://localhost:8501
```

---

# 🧪 Example Workflow

A typical workflow looks like this:

### Step 1 — Upload a dataset

Upload a CSV or Excel dataset.

### Step 2 — Explore the dataset

Review:

* Dataset dimensions
* Data types
* Missing values
* Basic statistics

### Step 3 — Perform EDA

Use:

* Correlation analysis
* Distribution plots
* Outlier detection
* Feature analysis

### Step 4 — Ask the LLM

Generate:

* Dataset summary
* Feature insights
* Machine Learning recommendations

### Step 5 — Chat with the AI

Ask natural-language questions about the dataset.

### Step 6 — Train Machine Learning models

Select:

```text
Target Feature
        +
Classification / Regression
```

and run the available models.

### Step 7 — Compare models

Review the evaluation metrics and determine which model performs best for the selected problem.

---

# 🎯 Project Goals

The main goal of this project is to develop an interactive **AI-assisted Data Science environment** that reduces the amount of repetitive work required during the early stages of a Data Science project.

The project focuses on combining:

```text
Traditional Data Science
        +
Machine Learning
        +
Large Language Models
        +
Interactive Visualization
```

into one application.

Rather than replacing the Data Scientist, the system is designed as a **copilot** that assists with exploration, interpretation, model selection, and experimentation.

---

# 🔮 Future Improvements

Potential future developments include:

* Automated target feature detection
* Automated problem-type detection
* Additional Machine Learning algorithms
* Hyperparameter optimization
* Cross-validation
* Feature importance analysis
* Automated feature engineering
* Model explainability
* Automated model selection
* Interactive prediction interface
* Exporting analysis reports
* Support for larger datasets
* More advanced LLM-based reasoning
* Persistent user projects and datasets

---

# 📚 Learning Objectives

This project was developed to explore and apply concepts from several areas of Data Science and Artificial Intelligence, including:

* Exploratory Data Analysis
* Statistical analysis
* Data visualization
* Feature analysis
* Data preprocessing
* Supervised Machine Learning
* Model evaluation
* Large Language Models
* Prompt engineering
* Conversational AI
* Streamlit application development
* Integration of AI into Data Science workflows

---

# 👨‍💻 Author

**Amirali Ghanbarpour**

Computer Engineering → Data Science & Artificial Intelligence

This project was developed as a practical implementation of an AI-assisted Data Science workflow.

---

# 📄 License

This project is intended primarily for educational and research purposes.

Add an appropriate license to this repository if you plan to distribute or reuse the project publicly.

```

### One recommendation before you put this on GitHub

I would **not** call it simply an "Automated Data Scientist" in the README. Your project is more accurately an **AI-assisted Data Scientist / Data Scientist Copilot**, because the user still chooses the target and problem type and the application assists with the analysis and modeling.

For a professor, I think the strongest one-line description is:

> **An interactive AI-powered Data Science assistant that combines Exploratory Data Analysis, LLM-based insights, conversational AI, and Machine Learning model comparison in a unified Streamlit application.**

That describes what you've actually built without overselling it.
```

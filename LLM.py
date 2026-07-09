import json
from openai import OpenAI
import requests

def eda_scan(eda_results):

    eda_json = json.dumps(eda_results, indent=4)
    prompt = f"""
    You are an experienced Senior Data Scientist.

    Your task is to analyze the following EDA scan.
    Write a professional report including:

    1. Dataset overview

    2. Data quality issues

    3. Missing value analysis

    4. Correlation insights

    5. Possible risks

    6. Suggestions before machine learning

    7. outliers

    EDA Scan:
    {eda_json}
    """

    # client = OpenAI(
    #     api_key = """
    # )

    # response = client.responses.create(
    #     model = "",
    #     input = prompt
    # )

    # report = response.output_text
    # return report

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )
    report = response.json()["choices"][0]["message"]["content"]
    return report





def eda_scan_each_feature(each_feature_summary):

    eda_json = json.dumps(each_feature_summary, indent=4)
    prompt = f"""
You are an experienced Senior Data Scientist.

Your task is to analyze each feature individually using the provided EDA scan.
For every feature, write a concise, clear, and practical analysis.

For each feature, include:

1. Feature Summary
   - Min, Max, Mean, Median, Standard Deviation
   - Any notable patterns in the basic statistics

2. Missing Values
   - Number of missing values
   - Whether missingness is important
   - Recommended handling (drop, impute, ignore)

3. Outliers
   - Number of outliers and their indices
   - Whether outliers are severe
   - Recommended handling (remove, cap, transform)

4. Distribution Insights
   - Describe the distribution shape (normal, skewed, heavy-tailed, etc.)
   - Mention if scaling or transformation is needed

5. Correlation Insights
   - Summarize Spearman and Pearson correlations with all other features
   - Highlight strong positive or negative relationships
   - Mention potential multicollinearity risks

6. Preprocessing Suggestions
   - Scaling (standardization, normalization, robust scaling)
   - Outlier handling
   - Imputation strategy
   - Possible transformations (log, box-cox, clipping)

Keep the analysis focused on the feature itself. 
Do not include dataset-level summaries.

EDA Scan (feature-level summary):
{eda_json}
    """
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )
    report = response.json()["choices"][0]["message"]["content"]
    return report


def model_recommendation(each_feature_summary, eda_results):
    eda_json_1 = json.dumps(eda_results, indent=4)
    eda_json_2 = json.dumps(each_feature_summary, indent=4)
    prompt = f"""
You are an experienced Senior Data Scientist.

Your task is to generate modeling recommendations for this dataset using the provided 
dataset-level summary (Part 1) and feature-level insights (Part 2).

Write a clear, structured, and practical modeling guide that includes:

1. Problem Type Identification
   - Identify whether the dataset is for classification or regression.
   - Explain briefly why.

2. Baseline Models
   - Recommend simple baseline models to establish initial performance.
   - Explain what insights these baselines provide.

3. Advanced Models
   - Recommend more powerful models suitable for the dataset.
   - Consider outliers, skewness, correlations, feature distributions, and dataset size.
   - Explain why each model is appropriate.

4. Model Sensitivity Considerations
   - Identify models sensitive to scaling, outliers, skewness, or multicollinearity.
   - Provide guidance on how to handle these issues.

5. Feature Importance & Selection
   - Suggest methods for identifying important features.
   - Recommend feature selection or dimensionality reduction techniques if needed.

6. Evaluation Strategy
   - Recommend appropriate evaluation metrics.
   - Suggest cross-validation strategy.
   - Mention class imbalance handling if relevant.

7. Final Modeling Pipeline
   - Provide a step-by-step modeling workflow from baseline to final model comparison.

Use ONLY the following inputs:
Dataset Summary:
{eda_json_1}

Feature-Level Insights:
{eda_json_2}

    """
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )
    report = response.json()["choices"][0]["message"]["content"]
    return report


#----------------Phase 4----------------------------

def ai_answer(question , eda_results, each_feature_summary, chat_history):
    eda_json_1 = json.dumps(eda_results, indent=4)
    eda_json_2 = json.dumps(each_feature_summary, indent=4)
    question_json: str = json.dumps(question, indent=4)
    chat_history_json: str = json.dumps(chat_history, indent=4)
    prompt = f"""
You are an expert Senior Data Scientist and AI Data Analysis Assistant.

Your role is to help users understand their dataset, perform better analysis, prepare data, and make machine learning decisions.

You have access to:

Dataset-level analysis results.
Feature-level analysis results.
Previous conversation history.

Your responsibilities:

Answer questions about the dataset clearly and accurately.
Explain statistical patterns and relationships.
Interpret correlations, distributions, missing values, and outliers.
Recommend suitable preprocessing techniques.
Suggest feature engineering strategies.
Recommend machine learning models based on the dataset characteristics.
Explain advantages and disadvantages of different approaches.
Help the user understand why a specific technique is appropriate.

Important rules:

1. Use ONLY the provided dataset information.
2. Do not invent statistics, columns, relationships, or results.
3. If the provided information is insufficient, clearly state that more information is needed.
4. Explain concepts in a practical Data Science context.
5. Give recommendations with reasoning, not just answers.
6. Consider dataset size, feature types, missing values, outliers, correlations, skewness, and possible target variables when making recommendations.
7. Remember previous messages in the conversation and use them when relevant.

Use ONLY the following inputs:
Dataset Summary:
{eda_json_1}

Feature-Level Insights:
{eda_json_2}

History of the Questions:
{chat_history_json}

The question that user asked:
{question_json}

    """
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )
    report = response.json()["choices"][0]["message"]["content"]
    return report


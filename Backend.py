import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vesal\Downloads\Auto.csv")

def pearson_correlation(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df.corr(method = 'pearson')


def pearson_correlation_heatmap(df):
    corr_matrix = pearson_correlation(df)
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, cmap ="YlGnBu", annot=True, ax= ax)
    return fig

def spearman_correlation(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df.corr(method = 'spearman')

def spearman_correlation_heatmap(df):
    spearman_corr_matrix = spearman_correlation(df)
    fig, ax = plt.subplots()
    sns.heatmap(spearman_corr_matrix, cmap = "YlGnBu" , annot=True , ax = ax)
    return fig

def IQRR(df):
    list_of_features = df.columns
    for feature in list_of_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3-Q1
        lower_boundry = Q1 - 1.5 * IQR
        upper_boundry = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_boundry) | (df[feature] > upper_boundry)]
        index_of_outliers = outliers.index
    return index_of_outliers, outliers

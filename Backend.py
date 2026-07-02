import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vesal\Downloads\Auto.csv")

def pearson_correlation(df):
    corr = df.corr(method = 'pearson')
    return corr


def pearson_correlation_heatmap(df):
    corr_matrix = pearson_correlation(df)
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, cmap ="YlGnBu", annot=True, ax= ax)
    return fig

def spearman_correlation(df):
    correlation_matrix = df.corr(method = 'spearman')
    return correlation_matrix

def spearman_correlation_heatmap(df):
    spearman_corr_matrix = spearman_correlation(df)
    fig, ax = plt.subplots()
    sns.heatmap(spearman_corr_matrix, cmap = "YlGnBu" , annot=True , ax = ax)
    return fig


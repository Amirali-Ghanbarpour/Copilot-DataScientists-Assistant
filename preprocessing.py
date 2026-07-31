import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from modeling import train_testsplit

def divide_the_dataset_into_x_y(df, target_feature):
    y = df[target_feature]
    x= df.drop(columns = [target_feature])
    return x , y

def select_feature_type_numerical(df):
    numeric_features = df.select_dtypes(include = ['int64' , 'float64']).columns
    return numeric_features

def select_feature_type_categorical(df):
    categorical_features = df.select_dtypes(include = ['object']).columns
    return categorical_features


def median_imputer_missing_data_handling(df, numerical_features):
    for feature in numerical_features:
        median_imputation = df[feature].fillna(df[feature].median())
        df[feature] = median_imputation
    return df

def robust_scaler_for_numerical_features(df, numerical_features):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[numerical_features])
    scaled_dataframe = pd.DataFrame(scaled_data, columns = df[numerical_features].columns)
    return scaled_dataframe

def most_frequent_missing_data_handeling(df, categorical_features):
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_data = imputer.fit_transform(df[categorical_features])
    imputed_dataframe = pd.DataFrame(imputed_data, columns = df[categorical_features].columns)
    return imputed_dataframe

def one_hot_encoding(df, categorical_features):
    encoder = pd.get_dummies(df[categorical_features])
    return encoder

def set_session_state(target_feature, problem_type):
    if "target_feature" not in st.session_state:
        # st.session_state.target_feature = []
        # st.session_state.problem_type = []

        st.session_state.target_feature = target_feature
        st.session_state.problem_type = problem_type


def preprocessing_before_training_models(target_feature , numerical_features, categorical_features, uploaded_data):
    if target_feature in numerical_features:
        numerical_features_without_target_feature = numerical_features.drop(target_feature)
        categorical_features_without_target_feature = categorical_features.copy()
    
    else:
        categorical_features_without_target_feature = categorical_features.drop(target_feature)
        numerical_features_without_target_feature = numerical_features.copy()  

    # Missing_numerical_data_handeling
    
    if len(numerical_features_without_target_feature) > 0:
        missing_numerical_data_handeling = median_imputer_missing_data_handling(uploaded_data, numerical_features_without_target_feature)
        scaled_numerical_data = robust_scaler_for_numerical_features(missing_numerical_data_handeling, numerical_features_without_target_feature)

    else:
        scaled_numerical_data = pd.DataFrame(index=uploaded_data.index)

    if len(categorical_features_without_target_feature) > 0:
        imputed_categorical_features = most_frequent_missing_data_handeling(uploaded_data, categorical_features_without_target_feature)
        one_hotencoding = one_hot_encoding(imputed_categorical_features, categorical_features_without_target_feature)
    else:
        one_hotencoding = pd.DataFrame(index=uploaded_data.index)


    X_for_train = pd.concat([one_hotencoding, scaled_numerical_data], axis = 1)

    X = X_for_train
    if target_feature in categorical_features:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(uploaded_data[target_feature])
    else:
        y = uploaded_data[target_feature]

    train_test_splited = train_testsplit(X,y)

    return train_test_splited
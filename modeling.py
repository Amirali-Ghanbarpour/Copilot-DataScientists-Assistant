import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, Lasso
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error

def train_testsplit(X,y):
    X_train , X_test , y_train, y_test  = train_test_split(X , y , test_size=0.2 , random_state=1234)
    return X_train, X_test, y_train, y_test

def logistic_regression(X_train, X_test, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def random_forest_calssifer(X_train, X_test, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def svc_classifier(X_train, X_test, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def knn_classifier(X_train , X_test, y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def linear_models(X_train, X_test, y_train):
    model_linear_reg = LinearRegression()
    model_ridge = Ridge()
    model_huber_regressor = HuberRegressor()
    model_lasso = Lasso()

    model_linear_reg.fit(X_train, y_train)
    model_ridge.fit(X_train, y_train)
    model_huber_regressor.fit(X_train, y_train)
    model_lasso.fit(X_train, y_train)

    y_pred_linear_reg = model_linear_reg.predict(X_test)
    y_pred_ridge = model_ridge.predict(X_test)
    y_pred_huber_regressor = model_huber_regressor.predict(X_test)
    y_pred_lasso = model_lasso.predict(X_test)
    return {"linear_regression" : y_pred_linear_reg , "ridge" : y_pred_ridge , "huber_regressor" : y_pred_huber_regressor , "lasso" : y_pred_lasso}

def confusion_matrixx(y_test, y_pred):
    confusion_mat = confusion_matrix(y_test, y_pred)
    return confusion_mat

def classification_rep(y_test, y_pred):
    classification_repor = classification_report(y_test, y_pred)
    return classification_repor

def report_for_linear_models(y_test, y_pred):
    mae = mean_absolute_error(y_test , y_pred)
    mse = mean_squared_error(y_test , y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred )
    return {"mse": mse, "rmse": rmse, "r2" : r2 , "mae" : mae}


def metrics(y_pred , y_test):
    precision = precision_score(y_pred , y_test , average='macro')
    recall = recall_score(y_pred , y_test , average='macro')
    f1_scoree = f1_score(y_pred , y_test , average='macro')
    accuracy = accuracy_score(y_pred , y_test)
    return {"precision": precision , "recall": recall , "f1_scoree": f1_scoree , "accuracy": accuracy}

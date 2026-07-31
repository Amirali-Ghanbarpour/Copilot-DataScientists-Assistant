import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu 
from Backend import pearson_correlation
from Backend import pearson_correlation_heatmap
from Backend import spearman_correlation
from Backend import spearman_correlation_heatmap
from scipy.stats import gaussian_kde
from LLM import eda_scan
from LLM import eda_scan_each_feature
from LLM import model_recommendation
from LLM import ai_answer
from preprocessing import divide_the_dataset_into_x_y
from preprocessing import select_feature_type_numerical
from preprocessing import select_feature_type_categorical
from preprocessing import median_imputer_missing_data_handling
from preprocessing import robust_scaler_for_numerical_features
from preprocessing import most_frequent_missing_data_handeling
from preprocessing import one_hot_encoding
from sklearn.model_selection import train_test_split
from modeling import train_testsplit
from modeling import logistic_regression
from modeling import confusion_matrixx
from modeling import classification_rep
from sklearn.preprocessing import LabelEncoder
from modeling import random_forest_calssifer
from modeling import svc_classifier
from modeling import knn_classifier
from modeling import metrics
from preprocessing import set_session_state
from preprocessing import preprocessing_before_training_models
from modeling import linear_models
from modeling import report_for_linear_models


st.set_page_config(layout="wide")

#changing the Back-ground color of the whole page
st.markdown("""
            <style>
            .stApp {
                background-color: #F8F8F8;
            }
            </style>
            """ , unsafe_allow_html=True)



#changing the background color of the title and the subtitle

st.markdown("""
    <div class = "title-backcolor">
        <h1> &#128202; Copilot Datascientist Assistant </h1>
            <hr>
        <em> phase 1 -- upload your dataset to begin </em>    
    </div>
""", unsafe_allow_html=True)

#CSS for changing the background color of the title and the subtitle

st.markdown("""
    <style>
        .title-backcolor {
            background-color: #FFFFFF;
            width: 100%;
            text-align: center;
            padding: 30px;
            margin-top: -49px;

        }
        .title-backcolor h1 {
            color: Black;
        }
        .title-backcolor em {
            font-size : 1.575rem;
            }

    </style>
""", unsafe_allow_html=True)


#--------------------Phase 3 Part 1-----------------------------------------------

#the upload box
st.space()
st.subheader("Upload your dataset (CSV or Excel)")

uploaded_data = st.file_uploader(label = "Drag and drop file here")

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
        "container": {"padding": "0!important", "background-color": "#e5f4ff"},
        "icon": {"font-size": "2px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "font-family":"Times New Roman" , "margin":"0px", "--hover-color": "#DBDBDB"},
        "nav-link-selected": {"background-color": "#278EF5"},
    }
        



    )

    if selected == "Developing Stats":
        
        st.subheader("Dataset Preview")
        st.write("First 5 rows of your Dataset:")
        st.write(uploaded_data.head(5))

        column1, column2, column3 = st.columns(3)
        with column1:
            
            st.subheader("Dataset Summary")
            rows , columns = uploaded_data.shape
            df_created_for_shape_inversion = pd.DataFrame({
                "Metric" : ["Rows" , "Columns"],
                "Value" : [rows , columns]
            })
            st.table(df_created_for_shape_inversion)

        
        with column2:
            container1 = st.container(border=True)
            container1.write("Columns Info")
            container1.write(uploaded_data.dtypes)

        with column3:
            st.write("Missing Values")
            uploaded_data = pd.DataFrame(uploaded_data)
            missing_mask = uploaded_data.isnull().sum()
            st.write(missing_mask)
        
            st.subheader("Basic Stats")
            column4, column5, column6 = st.columns(3)
            with column4:
                st.write("Mean")
                for feature in numerical_features:
                    st.write( feature , ":" , uploaded_data[feature].mean())
            with column5:
                st.write("Max")
                for feature in numerical_features:
                    st.write( feature , ":" , uploaded_data[feature].max())
            with column6:
                st.write("Min")
                for feature in numerical_features:
                    st.write(feature , ":" , uploaded_data[feature].min())


    
    if selected == "Correlation Fields":

        st.subheader("Pearson Correlation Matrix")
        # pearson_corr = pearson_correlation(uploaded_data)
        numerical_df = uploaded_data.select_dtypes(include=['int64', 'float64'])
        st.write(numerical_df.corr(method = 'pearson'))
        st.subheader("Pearson Correlation Heatmap")
        pearson_corr_heatm = pearson_correlation_heatmap(uploaded_data)
        st.pyplot(pearson_corr_heatm)

        st.subheader("Spearman's Correlation")
        # spearman_correlation_matrix = spearman_correlation(uploaded_data)
        st.write(numerical_df.corr(method='spearman'))
        st.subheader("Spearman's Heatmap")
        spearman_corr_heatm = spearman_correlation_heatmap(uploaded_data)
        st.pyplot(spearman_corr_heatm)

        st.subheader("Missing Values")
        st.subheader("Missing Values by number")
        uploaded_data = pd.DataFrame(uploaded_data)
        missing_data_in_dataset = uploaded_data.isnull().sum()
        st.write(missing_data_in_dataset)
        st.subheader("Missing Values Percentage (%)")
        num_data_in_dataset = len(uploaded_data)
        missing_percentage = (missing_data_in_dataset / num_data_in_dataset) * 100
        st.write(missing_percentage)
        # missing_percentage_coverted_to_str = str(missing_percentage)
        # missing_percentage_str_with_string_added = missing_percentage_coverted_to_str + '%'
        # new_dataframe = pd.DataFrame(missing_percentage_str_with_string_added, columns=["Missing"])
        # st.write(new_dataframe)
        
        
        #Barchart
        list_of_features = uploaded_data.columns
        fig2, ax2 = plt.subplots()
        plt.bar( list_of_features , missing_data_in_dataset)
        st.pyplot(fig2)

        #describing the dataframe
        st.header("Describing the dataset")
        st.write(uploaded_data.describe())    


    if selected == "Distributions":
        
        #Histogram
        list_of_features = uploaded_data.columns
        for feature in list_of_features:
            fig3, ax = plt.subplots()
            sns.histplot(uploaded_data[feature],
                kde=False,
                color='red',
                bins=30,
                ax = ax)
            st.pyplot(fig3)
        
        #KDE             
        for feature in numerical_features:
            fig4, ax = plt.subplots()
            sns.kdeplot(uploaded_data[feature], shade = True, ax = ax)
            st.pyplot(fig4)
        
        #DistPlot
        for feature in numerical_features:
            fig4, ax = plt.subplots()
            sns.distplot(uploaded_data[feature] , ax = ax, bins = 5)
            st.pyplot(fig4)
        
    
    if selected == "Outliers Detection":

        #IQR
        st.header("IQR Outliers")
        for feature in numerical_features:

            Q1 = uploaded_data[feature].quantile(0.25)
            Q3 = uploaded_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_boundry = Q1 - 1.5 * IQR
            upper_boundry = Q3 + 1.5 * IQR
            outliers = uploaded_data[(uploaded_data[feature]<lower_boundry) | (uploaded_data[feature] > upper_boundry)]  
            index_of_outliers = outliers.index
            st.subheader(f"{feature} outliers : {outliers.shape[0]}")
            st.subheader("Value of the Outliers in the Dataset")
            st.write(outliers[feature])

        
    if selected == "Feature Analysis":

         #box plot
        st.header("Box Plot")
        for feature in numerical_features: 
            fig5, ax = plt.subplots()
            sns.boxplot(y = uploaded_data[feature] , ax = ax)
            st.pyplot(fig5)
        

        #Violin plot
        st.header("Violin Plot")
        for feature in numerical_features:
            fig6 , ax = plt.subplots()
            sns.violinplot(uploaded_data[feature], ax = ax)
            st.pyplot(fig6)


        #Scatter Plot
        st.header("Scatter Plot")
        feature1 = st.selectbox(
            'Select feature X',
            options = numerical_features
        )
        st.write(feature1)

        feature2 = st.selectbox(
            'Select Feature Y',
            options = numerical_features
        )
        st.write(feature2)

        feature1_data = uploaded_data[feature1]
        feature2_data = uploaded_data[feature2]
        fig7 , ax = plt.subplots()
        sns.scatterplot(x = feature1_data , y = feature2_data ,ax = ax)
        st.pyplot(fig7)
    

    if selected == "LLM":
        if st.button("Generate Dataset Summary"):
            output_of_LLM_phase3_part1 = eda_scan(eda_results)
            st.subheader(output_of_LLM_phase3_part1)

        if st.button("Generate Feature Insights"):
            output_of_LLM_phase3_part2 = eda_scan_each_feature(each_feature_summary)
            st.subheader(output_of_LLM_phase3_part2)
        
        if st.button("Machine Learning Model Suggestion"):
            output_of_LLM_phase3_part3 = model_recommendation(each_feature_summary , eda_results)
            st.subheader(output_of_LLM_phase3_part3)

    
    if selected == "Chat with AI":
        question = st.chat_input("Ask our AI")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "eda_results" not in st.session_state:
            st.session_state.eda_results = []

        if "each_feature_summary" not in st.session_state:
            st.session_state.each_feature_summary = []

        st.session_state.eda_results = eda_results
        st.session_state.each_feature_summary = each_feature_summary

        if question:
            st.session_state.chat_history.append(
                {"role" : "user" , "content" : question}
            )
            answer = ai_answer(question, st.session_state.eda_results, st.session_state.each_feature_summary, st.session_state.chat_history)
            st.session_state.chat_history.append(
                {"role" : "AI","content" : answer}
            )

            for index, message in enumerate(st.session_state.chat_history, start = 1):        
                
                st.caption(f"Message{index}:")
                with st.chat_message(message["role"]):
                    st.write(message["content"])


    if selected == "AI Assistant":
        target_feature = st.selectbox('**select which feature is going to be your target**',
     (uploaded_data.columns))
        
        problem_type = st.radio("**select problem type**", ('Classification', 'Regression'))

        if problem_type == 'Classification':

            set_session_state(target_feature, problem_type)

            button_run_machine_learning_algorithms = st.button("Run Machine Learning")
            uploaded_data = uploaded_data.dropna(subset = [target_feature])
            

            train_test_splited = preprocessing_before_training_models(target_feature , numerical_features, categorical_features, uploaded_data)

            #Logistic_Regression
            evaluation_metrics_for_logistic_regression = 0
            # st.header("Logistic Regression")
            if problem_type == "Classification" and pd.api.types.is_float_dtype(uploaded_data[target_feature]):
                st.warning("Warning: Logistic Regression cannot have float labels, It is better to choose another feature as the target feature")

            else:
                logistic_reg_y_pred = logistic_regression(train_test_splited[0], train_test_splited[1], train_test_splited[2])
                conf_matrix = confusion_matrixx(train_test_splited[3], logistic_reg_y_pred)
                cls_report = classification_rep(train_test_splited[3], logistic_reg_y_pred)
                evaluation_metrics_for_logistic_regression = metrics(logistic_reg_y_pred , train_test_splited[3])

            #Random Forest Classifier
            evaluation_metrics_for_Randon_Forest = 0
            # st.header("RandomForest")
            if problem_type == 'Classification' and pd.api.types.is_float_dtype(uploaded_data[target_feature]):
                st.warning("Warning: Random Forest Classifier cannot have float labels, It is better to choose another feature as the target feature")
            else:
                random_forest_classifier_y_pred = random_forest_calssifer(train_test_splited[0], train_test_splited[1] , train_test_splited[2])
                conf_matrix = confusion_matrixx(train_test_splited[3], random_forest_classifier_y_pred)
                cls_report = classification_rep(train_test_splited[3] , random_forest_classifier_y_pred)
                evaluation_metrics_for_Randon_Forest = metrics(random_forest_classifier_y_pred , train_test_splited[3])

            # SVC Classifier
            evaluation_metrics_for_SVC = 0
            # st.header("SVC")
            if problem_type == 'Classification' and pd.api.types.is_float_dtype(uploaded_data[target_feature]):
                st.warning("Warning: SVC Classifier cannot have float labels, It is better to choose another feature as the target feature")

            else:
                svc_classifier_y_pred = svc_classifier(train_test_splited[0] , train_test_splited[1], train_test_splited[2])
                conf_matrix = confusion_matrixx(train_test_splited[3] , svc_classifier_y_pred)
                cls_report = classification_rep(train_test_splited[3] , svc_classifier_y_pred)
                evaluation_metrics_for_SVC = metrics(svc_classifier_y_pred , train_test_splited[3])


            #KNN_Classifier
            evaluation_metrics_for_KNN = 0
            # st.header("KNN_Classifier")
            if problem_type == 'Classification' and pd.api.types.is_float_dtype(uploaded_data[target_feature]):
                st.warning("Warning: KNN Classifier cannot have float labels, It is better to choose another feature as the target feature")                
            else:
                KNN_classifier_y_pred = knn_classifier(train_test_splited[0] , train_test_splited[1] , train_test_splited[2])
                conf_matrix = confusion_matrixx(train_test_splited[3] , KNN_classifier_y_pred)
                cls_report = classification_rep(train_test_splited[3] , KNN_classifier_y_pred)
                evaluation_metrics_for_KNN = metrics(KNN_classifier_y_pred , train_test_splited[3])

            #-----------comparing the classification models------------------
            if button_run_machine_learning_algorithms:
                st.header("comparing the classification models")

                conf_matrix = pd.DataFrame(
                {
                "Precision": [evaluation_metrics_for_logistic_regression["precision"], evaluation_metrics_for_Randon_Forest["precision"], evaluation_metrics_for_SVC["precision"], evaluation_metrics_for_KNN["precision"]],
                "Recall": [evaluation_metrics_for_logistic_regression["recall"], evaluation_metrics_for_Randon_Forest["recall"], evaluation_metrics_for_SVC["recall"], evaluation_metrics_for_KNN["recall"]],
                "F1-score": [evaluation_metrics_for_logistic_regression["f1_scoree"], evaluation_metrics_for_Randon_Forest["f1_scoree"], evaluation_metrics_for_SVC["f1_scoree"], evaluation_metrics_for_KNN["f1_scoree"]],
                "Accuracy": [evaluation_metrics_for_logistic_regression["accuracy"], evaluation_metrics_for_Randon_Forest["accuracy"], evaluation_metrics_for_SVC["accuracy"], evaluation_metrics_for_KNN["accuracy"]],
                },
                index=["Logistic_Regression", "Random Forest", "SVC", "KNN"],
                )
                st.table(conf_matrix)


        if problem_type == 'Regression':

            set_session_state(target_feature, problem_type)

            button_run_machine_learning_algorithms = st.button("Run Machine Learning")
            uploaded_data = uploaded_data.dropna(subset = [target_feature])
            train_test_splited = preprocessing_before_training_models(target_feature , numerical_features, categorical_features, uploaded_data)

            #Linear Regression
            evaluation_report_linear_models_linear_reg = 0
            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(uploaded_data[target_feature]):
                st.warning("Regression Models cannot handle target features with categorical values")
            else:
                linear_reg_y_pred = linear_models(train_test_splited[0] , train_test_splited[1] , train_test_splited[2])
                evaluation_report_linear_models_linear_reg = report_for_linear_models(train_test_splited[3] , linear_reg_y_pred["linear_regression"])

                # st.write("Linear Reg Evaluation:" , "MSE:", evaluation_report_linear_models_linear_reg["mse"], "RMSE:" , evaluation_report_linear_models_linear_reg["rmse"] , "R2:" , evaluation_report_linear_models_linear_reg["r2"])


            evaluation_report_linear_models_ridge = 0

            #Ridge Regression
            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(uploaded_data[target_feature]):
                st.warning("Regression Models cannot handle target features with categorical values")
            else:
                ridge_regressor_y_pred = linear_models(train_test_splited[0] , train_test_splited[1] , train_test_splited[2])
                evaluation_report_linear_models_ridge = report_for_linear_models(train_test_splited[3] , ridge_regressor_y_pred["ridge"])

                # st.write("Ridge Regression Evaluation:" , "MSE:", evaluation_report_linear_models_ridge["mse"], "RMSE:" , evaluation_report_linear_models_ridge["rmse"] , "R2:" , evaluation_report_linear_models_ridge["r2"])

            
            #Lasso Regression
            evaluation_report_linear_models_lasso = 0

            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(uploaded_data[target_feature]):
                st.warning("Regression Models cannot handle target features with categorical values")
            else:
                lasso_regressor_y_pred = linear_models(train_test_splited[0] , train_test_splited[1] , train_test_splited[2])
                evaluation_report_linear_models_lasso = report_for_linear_models(train_test_splited[3] , lasso_regressor_y_pred["lasso"])

                # st.write("Lasso Reg Evaluation:" , "MSE:", evaluation_report_linear_models_lasso["mse"], "RMSE:" , evaluation_report_linear_models_lasso["rmse"] , "R2:" , evaluation_report_linear_models_lasso["r2"])


            #Huber Regressor
            evaluation_report_linear_models_huber = 0
            if problem_type == 'Regression' and not pd.api.types.is_numeric_dtype(uploaded_data[target_feature]):
                st.warning("Regression Models cannot handle target features with categorical values")
            else:
                huber_regressor_y_pred = linear_models(train_test_splited[0] , train_test_splited[1] , train_test_splited[2])
                evaluation_report_linear_models_huber = report_for_linear_models(train_test_splited[3] , huber_regressor_y_pred["huber_regressor"])

                # st.write("Huber Reg Evaluation:" , "MSE:", evaluation_report_linear_models_huber["mse"], "RMSE:" , evaluation_report_linear_models_huber["rmse"] , "R2:" , evaluation_report_linear_models_huber["r2"])
        
            if button_run_machine_learning_algorithms:
                st.header("comparing the classification models")

                conf_matrix = pd.DataFrame(
                {
                "MSE": [evaluation_report_linear_models_linear_reg["mse"], evaluation_report_linear_models_ridge["mse"], evaluation_report_linear_models_huber["mse"] , evaluation_report_linear_models_lasso['mse']],
                "Rmse": [evaluation_report_linear_models_linear_reg["rmse"], evaluation_report_linear_models_ridge["rmse"], evaluation_report_linear_models_huber["rmse"] , evaluation_report_linear_models_lasso['rmse']],
                "R2": [evaluation_report_linear_models_linear_reg["r2"], evaluation_report_linear_models_ridge["r2"], evaluation_report_linear_models_huber["r2"] , evaluation_report_linear_models_lasso['r2']],
                "MAE": [evaluation_report_linear_models_linear_reg["mae"] , evaluation_report_linear_models_ridge['mae'] , evaluation_report_linear_models_huber['mae'] , evaluation_report_linear_models_lasso['mae']]

                },
                index=["Linear Regression", "Ridge Regression", "Huber Regressor" , "Lasso Regression"],
                )
                st.table(conf_matrix)

    


        

    







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

    #calculating IQR for dictionary input:
    list_of_features = uploaded_data.columns
    dict_of_outliers_per_feature = {}
    for feature in list_of_features:
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
    eda_results["Min"] = uploaded_data.min().to_dict()
    eda_results["Max"] = uploaded_data.max().to_dict()
    eda_results["Mean"] = uploaded_data.mean().to_dict()
    eda_results["Pearson Correlation"] = uploaded_data.corr("pearson").to_dict()
    eda_results["Spareman Correlation"] = uploaded_data.corr("spearman").to_dict()
    eda_results["Outliers"] = dict_of_outliers_per_feature

    # st.write(eda_results)

    #--------------Phase 3 part 2-------------------------------------------
    #Note: all the int,float outputs needed to be wrapped inside a float(), int() functions because Json file only accepts data from dictionary which are float64,int64
    each_feature_summary = {}
    for feature in list_of_features:
        each_feature_summary[feature] = {
            "Min" : float(uploaded_data[feature].min()),
            "Max" : float(uploaded_data[feature].max()),
            "Mean" : float(uploaded_data[feature].mean()),
            "Number of Missing Values" : int(uploaded_data[feature].isnull().sum()),
            "Median" : float(uploaded_data[feature].median()),
            "Standard Deviation" : float(uploaded_data[feature].std()),
            "Feature Description" : uploaded_data[feature].describe().to_dict(),
            "Outliers" : dict_of_outliers_per_feature[feature],
            "Spearman Correlation": {
                other_feature : uploaded_data[feature].corr(uploaded_data[other_feature] , method='spearman')
                for other_feature in list_of_features
            },
            
            "Pearson Correlation" : {
                other_feature : uploaded_data[feature].corr(uploaded_data[other_feature] , method='pearson')
                for other_feature in list_of_features
            }

        }
    
    #st.write(each_feature_summary)
    #-----------------------------------------------------------------
    #Option Menu 
    selected = option_menu(
        menu_title = None,
        options=["Developing Stats" , "Correlation Fields" , "Distributions" , "Outliers Detection" , "Feature Analysis" , "LLM" , "Chat with AI"],
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
            st.write(uploaded_data.mean())
        with column5:
            st.write("Max")
            st.write(uploaded_data.max())
        with column6:
            st.write("Min")
            st.write(uploaded_data.min())
    

    
    if selected == "Correlation Fields":

        st.subheader("Pearson Correlation Matrix")
        pearson_corr = pearson_correlation(uploaded_data)
        st.write(pearson_corr)
        st.subheader("Pearson Correlation Heatmap")
        pearson_corr_heatm = pearson_correlation_heatmap(uploaded_data)
        st.pyplot(pearson_corr_heatm)

        st.subheader("Spearman's Correlation")
        spearman_correlation_matrix = spearman_correlation(uploaded_data)
        st.write(spearman_correlation_matrix)
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
        for feature in list_of_features:
            fig4, ax = plt.subplots()
            sns.kdeplot(uploaded_data[feature], shade = True, ax = ax)
            st.pyplot(fig4)
        
        #DistPlot
        for feature in list_of_features:
            fig4, ax = plt.subplots()
            sns.distplot(uploaded_data[feature] , ax = ax, bins = 5)
            st.pyplot(fig4)
        
    
    if selected == "Outliers Detection":

        #IQR
        st.header("IQR Outliers")
        for feature in list_of_features:
            Q1 = uploaded_data[feature].quantile(0.25)
            Q3 = uploaded_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_boundry = Q1 - 1.5 * IQR
            upper_boundry = Q3 + 1.5 * IQR
            outliers = uploaded_data[(uploaded_data[feature]<lower_boundry) | (uploaded_data[feature] > upper_boundry)]  
            index_of_outliers = outliers.index
            st.subheader(f"{feature} outliers : {outliers.shape[0]}")
            st.subheader("Line of the Outliers in the Dataset")
            st.write(index_of_outliers)

        
    if selected == "Feature Analysis":

         #box plot
        st.header("Box Plot")
        list_of_features = uploaded_data.columns
        for feature in list_of_features: 
            fig5, ax = plt.subplots()
            sns.boxplot(y = uploaded_data[feature] , ax = ax)
            st.pyplot(fig5)
        

        #Violin plot
        st.header("Violin Plot")
        list_of_features = uploaded_data.columns
        for feature in list_of_features:
            fig6 , ax = plt.subplots()
            sns.violinplot(uploaded_data[feature], ax = ax)
            st.pyplot(fig6)


        #Scatter Plot
        st.header("Scatter Plot")
        list_of_features = uploaded_data.columns
        feature1 = st.selectbox(
            'Select feature X',
            options = list_of_features
        )
        st.write(feature1)

        feature2 = st.selectbox(
            'Select Feature Y',
            options = list_of_features
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


            

           


        
        
            

    


        

    











# # st.set_page_config(layout="wide")
# # st.title("Copilot Datascientist Assistant" , text_alignment="center")
# # st.divider()
# # st.subheader("phase 1 -- upload your dataset to begin" , text_alignment="center")



# selected = option_menu(
#         menu_title = None,
#         options=["Home" , "Projects" , "Contact"],
#         orientation = "horizontal",
#     )

# if selected == "Home":
#     st.title(f"you have selected {selected}")
    
# if selected == "Projects":
#     st.title(f"you have selected {selected}")
    
# if selected == "Contact":
#     st.title(f"you have selected {selected}")
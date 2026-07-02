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


#the upload box
st.space()
st.subheader("Upload your dataset (CSV or Excel)")

uploaded_data = st.file_uploader(label = "Drag and drop file here")


# the data manipulations on the dataset
if uploaded_data:
    uploaded_data = pd.read_csv(uploaded_data)
    
    #Option Menu 
    selected = option_menu(
        menu_title = None,
        options=["Developing Stats" , "Correlation Fields" , "Distributions" , "Outliers Detection" , "Feature Analysis"],
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
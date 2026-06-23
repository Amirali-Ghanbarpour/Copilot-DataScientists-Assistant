import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu 


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
        <h1> &#128202; Copilo Datascientist Assistant </h1>
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
        options=["Developing Stats" , "Correlation Fields" , "Distributions"],
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

        st.subheader("Dataset Summary")
        rows , columns = uploaded_data.shape
        st.write("Shape")
        st.write("Rows:" , rows)
        st.write("Columns:" , columns)

        st.divider()
        st.write("Columns Info")
        st.write(uploaded_data.dtypes)

        st.divider()
        st.write("Missing Values")
        uploaded_data = pd.DataFrame(uploaded_data)
        missing_mask = uploaded_data.isnull().sum()
        st.write(missing_mask)

        st.divider()
        st.subheader("Basic Stats")
        st.write("Mean")
        st.write(uploaded_data.mean())
        st.write("Max")
        st.write(uploaded_data.max())
        st.write("Min")
        st.write(uploaded_data.min())
    

    
    if selected == "Correlation Fields":
        None
    

    if selected == "Distributions":
        None

    











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
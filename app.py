from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
#import pandas_profiling
from ydata_profiling import ProfileReport
import pandas as pd
#from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder
import os 
from pycaret.classification import setup, compare_models, pull, save_model, load_model  # Import for classification
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, pull as regression_pull, save_model as regression_save_model, load_model as regression_load_model 

def encode_columns(df, enselect):
  for col in enselect:
    df[col] = LabelEncoder().fit_transform(df[col])
  return df


with st.sidebar: 
    st.image("/workspaces/AutoStreamlit/download.jpeg")
    st.title("AutoCharanML")
    choice = st.radio("Navigation", ["Upload","Profiling","Cleaning","Modelling", "Download"])
    st.info("Welcome to AutoCharanML, the ultimate platform for discovering the best AutoML algorithms. Our site streamlines the process of evaluating and comparing AutoML solutions, giving you the insights needed to choose the most effective tool for your needs. With our intuitive benchmarks and expert guidance, finding the ideal AutoML algorithm has never been easier. Start optimizing your machine learning workflows with AlgoVista today!")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
# change start
if 'df' not in st.session_state:
    st.session_state['df']=None

#def set_data(current_df):
    #global df
    #df = current_df
# change end

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    #set_data(pd.read_csv('dataset.csv',index_col=None))


if choice == "Profiling":
    st.title("Upload Your Dataset")
    profile = ProfileReport(df, title='Pandas Profiling Report')
    st.markdown(profile.to_html(), unsafe_allow_html=True)

if choice == "Cleaning":
    st.title('Cleaning Categorical Data')
    st.write("In this section, we will perform essential data preprocessing steps on categorical data, including deletion of irrelevant columns and encoding categorical features into numerical format. These transformations are crucial for building robust machine learning models and ensuring optimal application performance.")
    st.write("The following table presents the initial five rows of the dataset to provide a preliminary overview of the data structure and content.")
    col_info=df.head(5)
    st.write(col_info)
    x=df.dtypes
    st.write("To initiate data cleaning, we will examine the data types of each column to identify potential inconsistencies and inform subsequent preprocessing steps")
    st.write(x)
    tar=st.multiselect('Please select the columns you wish to remove from the dataset. These columns will be excluded from further analysis',df.columns)
    if tar:
        if st.button('Delete'):
            df=df.drop(columns=tar,axis=1)
            st.success(f"Columns{tar} Deleted Succesfully")
            st.dataframe(df.head())
            st.session_state['df']=df
            
        st.title("Encoding")
        st.write('Encoding is a critical preprocessing step that converts categorical data into numerical representations, enabling machine learning algorithms to effectively process and analyze information')
        df=st.session_state['df']
        enselect=st.multiselect('Identify the categorical columns that require transformation into numerical format for subsequent machine learning modeling',df.columns)
        #st.dataframe(df.head())
        if st.button('Encode'):
            if enselect:
                for col in enselect:
                    df = encode_columns(df, enselect)
                  #set_data(encode_columns(df,enselect))
                st.write(df.head())
                st.session_state['df']=df
            else:
                st.warning("Please select at least one categorical column to proceed with the encoding process.")
    
    else:
        st.title("Encoding")
        st.write('Encoding is a critical preprocessing step that converts categorical data into numerical representations, enabling machine learning algorithms to effectively process and analyze information')
        enselect=st.multiselect('Identify the categorical columns that require transformation into numerical format for subsequent machine learning modeling',df.columns)
        #st.dataframe(df.head())
        if st.button('Encode'):
            if enselect:
                for col in enselect:
                    df = encode_columns(df, enselect)
                  #set_data(encode_columns(df,enselect))
                st.write(df.head())
                st.session_state['df']=df
            else:
                st.warning("Please select at least one categorical column to proceed with the encoding process")


if choice == "Modelling":
  df = st.session_state['df']
  if st.button('Check Dataset'):
    st.write(df.head())
  # Choose between classification or regression
  st.write("Classification predicts categorical outcomes (yes/no, spam/not spam), while regression predicts continuous values (house prices, sales figures). Choose classification for categorical targets and regression for numerical ones.")
  task_type = st.selectbox("Select Task Type", ["Classification", "Regression"])

  

  if task_type == "Classification":
    chosen_target = st.selectbox('Choose the Target Column (Classification)', df.columns)
    if st.button('Run Modelling'):
      setup(df, target=chosen_target, task_type="classification")  # Specify classification task_type
      setup_df = pull()
      st.dataframe(setup_df)
      best_model = compare_models()
      compare_df = pull()
      st.dataframe(compare_df)
      save_model(best_model, 'best_model')

  else:
    chosen_target = st.selectbox('Choose the Target Column (Regression)', df.columns)
    if st.button('Run Modelling'):
      regression_setup(df, target=chosen_target)  # Use regression_setup for regression
      setup_df = regression_pull()
      st.dataframe(setup_df)
      best_model = regression_compare_models()
      compare_df = regression_pull()
      st.dataframe(compare_df)
      regression_save_model(best_model, 'best_model')



if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")





    
    

                 
                 
    
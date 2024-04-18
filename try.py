import streamlit as st
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model,predict_model
from pandasai import SmartDataframe as sd
from pandasai.llm import GooglePalm
import ydata_profiling as pp
import pandas as pd
import pickle
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

os.environ["PANDASAI_API_KEY"]="$2a$10$BZhVRP9.5P6AS4Xf0caO3ubsFSE0cz1H1q/Hzxzl3SHMUInxUron6"

chosen_target = ''

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("LeytonAnalysis")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

        prompt = st.text_area("Enter your prompt")
        if st.button("Query"):
            if prompt:
                with st.spinner("Generating response ..."):
                    llm = GooglePalm(api_key="AIzaSyAzNWh54dprGIBBLBWocwbFKsxTeualqQU")
                    sdf = sd(df,config={"llm":llm})
                    response = sdf.chat(prompt)
                    st.success(response)
                    st.set_option('deprecation.showPyplotGlobalUse',False)
                    st.pyplot()
            else:
                st.warning("Please enter a prompt")


if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    model_type = st.selectbox("Choose model types",["Regression","Classification"])
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models(include=['lr','rf'])
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')


if choice == "Downaload": 
    # with open('best_model.pkl', 'rb') as f: 
    #     model = load_model('./best_model.pkl')
    #     chosen_target = st.selectbox('Choose the Target Column', df.columns)
    #     st.dataframe(df.head(3))
    #     if chosen_target:
    #         y_values = df[chosen_target]
    #         x_values = df = df.drop(chosen_target, axis=1)
    #         simple_ar = []
    #         for x in x_values:
    #                 name = st.text_input(x,value="")
    #                 simple_ar.append(name)
    #         if st.button("Predict"):
    #             predicted_value = predict_model(model,[simple_ar])

    #             st.text(predicted_value)

        st.download_button('Download Model', f, file_name="best_model.pkl")
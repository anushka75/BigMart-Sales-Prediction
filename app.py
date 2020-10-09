import numpy as np
import pandas as pd
import streamlit as st

import pickle



pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)

def welcome():
    return "Welcome All"


def predict_sales_prediction(Item_Weight,Fat_Content,Visibility,MRP,Establishment_year,Size,Location_type,Outlet_type,Item_Combined,Outlet_Code):
    
    """Let's Predict The Sales Of Outlets
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Item_Weight
        in: query
        type: number
        required: true
      - name: Fat_Content
        in: query
        type: number
        required: true
      - name: Visibility
        in: query
        type: number
        required: true
      - name: MRP
        in: query
        type: number
        required: true
      - name: Establishment_year
        in: query
        type: number
        required: true
      - name: Size
        in: query
        type: number
        required: true
      - name: Item_Combined
        in: query
        type: number
        required: true
      - name: Outlet_type
        in: query
        type: number
        required: true
      - name: Location_type
        in: query
        type: number
        required: true
      - name: Outlet_Code
        in: query
        type: number
        required: true
    responses:
        200:
            description: The prediction is
        
    """
   
    prediction=model.predict([[Item_Weight,Fat_Content,Visibility,MRP,Establishment_year,Size,Location_type,Outlet_type,Item_Combined,Outlet_Code]])
    print(prediction)
    return prediction



def main():
    #st.title("BigMart Sales Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">BigMart Sales Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Item_Weight = st.text_input("Item Weight","Type Here")
    Fat_Content = st.text_input("Item Fat Content","Type Here")
    Visibility= st.text_input("Item Visibility","Type Here")
    MRP = st.text_input("Item MRP","Type Here")
    Establishment_year = st.text_input("Outlet Establishment Year","Type Here")
    Size = st.text_input("Outlet Size","Type Here")
    Location_type = st.text_input("Outlet Location Type","Type Here")
    Outlet_type = st.text_input("Outlet Type","Type Here")
    Item_Combined = st.text_input("Item Combined","Type Here")
    Outlet_Code = st.text_input("Outlet Code","Type Here")
    
    result=""
    if st.button("Predict"):
        result=predict_sales_prediction(Item_Weight,Fat_Content,Visibility,MRP,Establishment_year,Size,Location_type,Outlet_type,Item_Combined,Outlet_Code)
    st.success('The prediction is {}'.format(result))
    if st.button("About"):
        st.text("This helps in predicting the sale of any product available in the outlet")

if __name__=='__main__':
    main()
    
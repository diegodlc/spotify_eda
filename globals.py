import pandas as pd
import streamlit as st

def initialize(): 
    global df
    df = load_data('./Spotify_Features.csv')

#TODO: borrar Nans


@st.cache(persist=True, show_spinner=True)
# Load  the Data 
def load_data(file):
    df = pd.read_csv(file)
    return df


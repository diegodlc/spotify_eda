import streamlit as st
import numpy as np
import pandas as pd


import pages.home 
import pages.eda
import pages.pca

import resources.ast as ast
import globals

# Initial page config

st.set_page_config(
     page_title='Spotify dataset exploration',
    #  layout="wide",
     initial_sidebar_state="expanded",
)

PAGES = {
    "Home": pages.home,
    "EDA" : pages.eda,
    "PCA" : pages.pca
}

def main():
    globals.initialize() 
    df = globals.df

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.write()


      


    # if st.button("Balloons"):
    #     st.balloons() 


    
    
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app was made by Diego de la Cal for the subject 'Large Scale Media Analytics' of the MSTC - UPM.
        [Github](https://github.com/diegodlc/spotify_eda)
        """
    )

 

if __name__ == "__main__":
    main()

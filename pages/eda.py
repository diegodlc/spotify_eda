import streamlit as st 
import globals
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
from pandas.plotting import scatter_matrix

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import altair as alt
import numpy as np
import pandas as pd
import time
import streamlit.components.v1 as components




sns.set(rc={'figure.figsize':(11.7,8.27)})
st.set_option('deprecation.showPyplotGlobalUse', False)

numerics = ['int64', 'float64']



def dibuja_progreso():
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart = st.line_chart(np.random.randn(10, 2))
    for i in range(50):
        # Update progress bar.
        progress_bar.progress(i + 1 )
        new_rows = np.random.randn(10, 2)
        # Update status text.
        status_text.text(
            'The latest random number is: %s' % new_rows[-1, 1])
        # Append data to the chart.
        chart.add_rows(new_rows)
        # Pretend we're doing some computation that takes time.
        time.sleep(0.1 )
    status_text.text('Done!')
    st.balloons()


def pairplot(df):

    # fig = sns.pairplot(df)
    # st.pyplot(fig)
    df

def compara_features(df):
    Types_of_Features=["key","time_signature","mode","tempo","valence","speechiness",
                      "loudness","liveness","instrumentalness","energy",
                      "danceability","acousticness"]
    Name_of_Feat = st.selectbox("Feature 1", Types_of_Features)
    Name_of_Feat2 = st.selectbox("Feature 2", Types_of_Features)



    # Sort_DF = df.sort_values(by=['liveness'], ascending=False)
    # chart_df = Sort_DF[Types_of_Features]
    chart_df = df[Types_of_Features]

    st.header(f'{Name_of_Feat.capitalize()} vs. {Name_of_Feat2.capitalize()}')
    c = alt.Chart(chart_df).mark_circle().encode(
        alt.X(f'{Name_of_Feat2}', scale=alt.Scale(zero=False)), y=f'{Name_of_Feat}', color=alt.Color(f'{Name_of_Feat2}', scale=alt.Scale(zero=False)), 
        size=alt.value(200), tooltip=["key","time_signature","mode"])

    st.altair_chart(c, use_container_width=True)






def explore_dataframe(df):
    features = st.multiselect(
        "Choose features", list(df.columns), ["danceability", "acousticness"]
    ) 
    if not features:
        st.error("Please select at least one feature.")
    else:
        data = df[features]
        chart = (
            alt.Chart(  )
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )

        st.altair_chart(chart, use_container_width=True)



def histograms(df):
    st.subheader('Histogram | Distplot')
    st.text("")
    x = df.select_dtypes(include=numerics) 
    column_dist_plot = st.selectbox("Select variable to explore.",x.columns)
    
    fig = sns.distplot(df[column_dist_plot])
    st.pyplot()

def boxplot(df):
    st.subheader('Boxplot')
    # column_box_plot_X = st.sidebar.selectbox("X (Choose a column). Try Selecting island:",df.columns.insert(0,None))
    # column_box_plot_Y = st.sidebar.selectbox("Y (Choose a column - only numerical). Try Selecting Body Mass",df.columns)
    # hue_box_opt = st.sidebar.selectbox("Optional categorical variables (boxplot hue)",df.columns.insert(0,None))
    # if st.checkbox('Plot Boxplot'):
    # fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y,data=df,palette="Set3")
    # st.pyplot()
    columns =["valence","speechiness","liveness","instrumentalness","energy","danceability","acousticness"]
    # fig2 = df.boxplot(column=columns,figsize=(16,10), grid=False)

    fig2 = sns.boxplot(x="variable", y="value", data=pd.melt(df[columns]))
    st.pyplot()

    st.subheader('Compare variables')
    x_cols = ['key', 'mode' ]
    column_box_plot_X = st.selectbox("X (Choose a categorical column)",x_cols)
    column_box_plot_Y = st.selectbox("Y (Choose a column - only numerical).",df.select_dtypes(include=numerics).columns)
    fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y,data=df,palette="Set3")
    st.pyplot()




def correlation(df):
    st.subheader('Correlation Matrix')
    corrmat = df.corr()
    plt.figure(figsize=(16, 13))
    sns.heatmap(corrmat, vmax=.8, square=True,annot=True,cmap="coolwarm")
    st.pyplot()
    
    # st.subheader('Top 10 most related features')
    # valid_cols=["key","valence","speechiness","loudness","liveness","instrumentalness","energy","danceability","acousticness"]
    # var = st.selectbox('Select feature',valid_cols )
    # k = 10 #number of variables for heatmap
    # cols = corrmat.nlargest(k, var)[var].index
    # cm = np.corrcoef(df[cols].values.T)
    # sns.set(font_scale=1.25)
    # sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    # st.pyplot()
    with st.beta_expander("Notes"):
        st.write("""
        We can obtain a lot of useful information. Fo instance, we can see that there is a high/medium correlation (0.68) 
        between energy feature and loudness which means a relationship between both features, something that is not surprising 
        at all since The more energy in a song, the louder that song is. 

        Besides, There are few other cases which we can highlight their correlation. The first one is the correlation between valence and danceability (0.43) which is pretty normal since valence represents a measure describing the musical positiveness conveyed by a song. So, songs with high valence sound more positive (e.g. happy, cheerful, euphoric) and otherwise in the case of low valence. Moreover, tracks which are more danceable tend to be more positive. The second case is correlation between energy and valence (0.31) which is expected from tracks that tend to be more positive and euphoric, but in this last case correlation is not high enough to assume a strong correlction between each other features. The last one is so interesting because of there is a negative correlation bewteen acousticness and energy (-0.51), which is reasonable since acoustic tracks tend to be more peaceful, quite, serene than high energy tracks which tend to be more euphoric and active, as we have assumed before.

        On the other hand, we can see the lack of correlation between features which could be correlated in a first instance. 
        For example, danceability and energy both have a correlation close to 0 despite both features could have been related 
        at first. However, both features are correlated with valence directly so might be a indirectly relation between both. 
        Other interesting point is the comparison between acousticness and loudness since both present a negative correlation, 
        which is reasonable since, normally, acosutic tracks usually be quiet and calm.

        """
        )
        




#================================
#       MAIN 
#================================

def write():
    df = globals.df
    
    st.title('Exploratory Data Analysis :books:')
    st.write("""

    Let's dive deeper on the dataset. In this page we can analye and compare our features to have a better understanding of each one. 
    Use the __Explore__ section of the sidebar to select which tool you want to use. 

    """)
    st.sidebar.title("Explore")
    if st.sidebar.checkbox('Histogram | Distplot'):
        histograms(df)

    if st.sidebar.checkbox('Boxplot'):
        boxplot(df)

    if st.sidebar.checkbox("Correlation Matrix"):
        correlation(df)



    # pairplot(df)
    # dibuja_progreso()


    # if st.checkbox("Generte Pandas Profiling"):
    #     pr = ProfileReport(df, explorative=True)
    #     st_profile_report(pr)

    # compara_features(df)

    #Ejemplo plot
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(
    #     df["liveness"],
    #     df["duration"],
    # )
    # ax.set_xlabel("Acceleration")
    # ax.set_ylabel("Miles per gallon")

    # st.write(fig)

    # st.header("test html import")

    # HtmlFile = open("/home/delak/Descargas/EDA(1).html", 'r', encoding='utf-8')
    # source_code = HtmlFile.read() 
    # # print(source_code)
    # components.html(source_code, height=600, width=800)

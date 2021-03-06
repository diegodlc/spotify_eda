import streamlit as st
import globals
import io
import missingno as msno



buf = io.StringIO()



def missing_values(df):
    st.subheader('Missing values')
    st.write("As we can see, there's a few missing rows on our dataset. Let's remove them with the button below.")
    if st.button("Remove NaNs"):
        df = df.dropna()
    # Visualize missing values as a matrix 
    p = msno.matrix(df, inline=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(p)
    #TODO: utilizar imputer para los NaNs


def write():
    
    df = globals.df
    

    col1, col2 = st.beta_columns([10,40])
    with col1:
        st.image('https://download.logo.wine/logo/Spotify/Spotify-Logo.wine.png', width=300)
    with col2:
        st.title('Exploratory Data Analysis  ')

    st.write('[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yR88TKbS3SbKr5wa2TSj_lm-gvMD3Jcb?usp=sharing)')
    st.write("""
    We will go through the [Spotify dataset](https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features) in order to gain some useful information. You can use the sidebar to navigate trhough the 
    different pages. 

    First, let's have a look at the dataset below. Use the sidebar to check the options you want to visualize and don't forget to get rid of the missing values.
    """)

    st.subheader('Raw Data')
    st.dataframe(df) 

    with st.beta_expander("See features description"):
        st.write("""
- __duration_ms:__ _(int)_ The duration of the track in miliseconds.
- __key:__ _(int)_ The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C#/Db, 2 = D, and so on. If no key was detected, the value is -1.
- __mode:__ _(int)_ Indicates the modelity (mayor or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- __time_signature:__ _(int)_ An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
- __acousticness:__ _(float)_ A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. __Acoustic music__ is music that solely or primarily usus instruments that produce sound through acousticmeans, as opposed to electric or electronic means. The distribution of values for this feature look like this: <img src="images/acousticness.png">
- __danceability:__	_(float)_ Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. The distribution of values for this feature look like this: <img src="images/danceability.png">
- __energy:__ _(float)_ Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. The distribution of values for this feature look like this: <img src="images/energy.png">
- __instrumentalness:__ _(float)_ Predicts whether a track contains no vocals. ???Ooh??? and ???aah??? sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly ???vocal???. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this: <img src="images/instrumentalness.png">
- __liveness:__ _(float)_ Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. The distribution of values for this feature look like this: <img src="images/liveness.png">
- __loudness:__ _(float)_ The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. The distribution of values for this feature look like this: <img src="images/loudness.png">
- __speechiness:__ _(float)_ Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. The distribution of values for this feature look like this: <img src="images/speechiness.png">
- __valence:__ _(float)_ A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). The distribution of values for this feature look like this: <img src="images/valence.png">
- __tempo:__ _(float)_ The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. The distribution of values for this feature look like this: <img src="images/tempo.png">
- __spotify_id:__ _(string)_ The Spotify ID for the track.


        """)
    st.text("")

    # if st.sidebar.checkbox('Show Raw data'):
    #     st.subheader('Raw Data')
    #     st.dataframe(df) 


    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)


    if st.sidebar.checkbox('Statistical Description'):
            st.subheader('Statistical Data Descripition')
            st.write(df.describe())

    if st.sidebar.checkbox("Missing values"):
        missing_values(df)


    st.info("After visualizing and fixing the missing values, let's continue the exploration on the 'EDA' page. ")
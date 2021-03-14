import streamlit as st
import globals

import numpy as np
import pandas as pd 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from kneed import KneeLocator
from matplotlib import cm



def PCA_sklearn(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] #seleccionar datos numéricos
    x = df[0:5].select_dtypes(include=numerics)     #0:5 pq no hay NaNs. #TODO: remove Nans
    pca = PCA(n_components=2)
    X_r = pca.fit(x).transform(x)
    # print(X_r)

@st.cache(allow_output_mutation=True)
def kmeans(df):
    features=["tempo","valence","speechiness","liveness","energy","danceability","acousticness"]
    df_X = df[features]
    print("Standard scaler and PCA")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_X) 
    pca = PCA()
    pca.fit(X_std)
    evr = pca.explained_variance_ratio_
    for i, exp_var in enumerate(evr.cumsum()):
        if exp_var >= 0.8:
            n_comps = i + 1
            break
    print("Finding optimal number of components", n_comps)
    pca = PCA(n_components=n_comps)
    pca.fit(X_std)
    scores_pca = pca.transform(X_std)
    wcss = []
    max_clusters = 11
    for i in range(1, max_clusters):
        kmeans_pca = KMeans(i, init='k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
    n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
    print("Finding optimal number of clusters", n_clusters)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1 = num_components_graph(ax1, len(df_X.columns), evr)
    # ax2 = num_clusters_graph(ax2, max_clusters, wcss)
    # fig.tight_layout()
    print("Performing KMeans")
    kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    df_seg_pca_kmeans = pd.concat([df_X.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_seg_pca_kmeans.columns.values[(-1 * n_comps):] = ["Component " + str(i+1) for i in range(n_comps)]
    df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_
    df['Cluster'] = df_seg_pca_kmeans['Cluster']
    df['Component 1'] = df_seg_pca_kmeans['Component 1']
    df['Component 2'] = df_seg_pca_kmeans['Component 2']
    return df, n_clusters


@st.cache(allow_output_mutation=True)
def make_normalized_df(df, col_sep):
    print(len(df))
    non_features = df[df.columns[:col_sep]]
    features = df[df.columns[col_sep:]]
    norm = MinMaxScaler().fit_transform(features)
    scaled = pd.DataFrame(norm, index=df.index, columns = df.columns[col_sep:])
    return pd.concat([non_features, scaled], axis=1)


@st.cache(allow_output_mutation=True)
def make_radar_chart(norm_df, n_clusters):
    fig = go.Figure()
    cmap = cm.get_cmap('tab20b')
    angles = list(norm_df.columns[5:])
    angles.append(angles[0])

    layoutdict = dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                ))
    maxes = dict()

    for i in range(n_clusters):
        subset = norm_df[norm_df['cluster'] == i]
        data = [np.mean(subset[col]) for col in angles[:-1]]
        maxes[i] = data.index(max(data))
        data.append(data[0])
        fig.add_trace(go.Scatterpolar(
            r=data,
            theta=angles,
            # fill='toself',
            # fillcolor = 'rgba' + str(cmap(i/n_clusters)),
            mode='lines',
            line_color='rgba' + str(cmap(i/n_clusters)),
            name="Cluster " + str(i)))
        
    fig.update_layout(
            polar=layoutdict,
            showlegend=True
    )
    fig.update_traces()
    return fig, maxes




@st.cache(allow_output_mutation=True)
def get_color_range(n_clusters):
    cmap = cm.get_cmap('tab20b')    
    range_ = []
    for i in range(n_clusters):
        color = 'rgb('
        mapped = cmap(i/n_clusters)
        for j in range(3):
            color += str(int(mapped[j] * 255))
            if j != 2:
                color += ", "
            else:
                color += ")"
        range_.append(color)
    return range_

def visualize_clusters(df, n_clusters, range_):
    graph = alt.Chart(df.reset_index()).mark_point(filled=True, size=60).encode(
        x=alt.X('Component 2'),
        y=alt.Y('Component 1'),
        # shape=alt.Shape('playlist', scale=alt.Scale(range=["circle", "diamond", "square", "triangle-down", "triangle-up"])),
        color=alt.Color('Cluster', scale=alt.Scale(domain=[i for i in range(n_clusters)], range=range_)),
        # tooltip=['name', 'artist']
    ).interactive()
    st.altair_chart(graph, use_container_width=True)


def write():
    st.title("Principal Component Analysis")
    df = globals.df
    df_original = df.copy()
    features=["tempo","valence","speechiness","liveness","instrumentalness","energy","danceability","acousticness"]
    df = df[features]
    df = df.dropna()
    

    # PCA_sklearn(df)

    # implement k-means clustering with PCA
    clustered_df, n_clusters = kmeans(df)

    # make radar chart to help understand the cluster differences
    cluster_labels = clustered_df['Cluster']
    orig = clustered_df.drop(columns=['Cluster', "Component 1", "Component 2"])
    orig.insert(4, "cluster", cluster_labels)
    norm_df = make_normalized_df(orig, 5)
    fig, maxes = make_radar_chart(norm_df, n_clusters)
    st.write(fig)

    # interactive visualizations of clusters on 2D plane
    range_ = get_color_range(n_clusters)
    visualize_clusters(clustered_df, n_clusters, range_)
    
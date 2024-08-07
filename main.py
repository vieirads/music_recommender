import streamlit as st
from recommender import *

import pandas as pd
import pickle

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

from collections import defaultdict

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from skimage import io


# functions
def find_song(artist, name):
    song_data = defaultdict()
    results = sp.search(q="artist: {} track: {}".format(artist, name), limit=1)
    if results["tracks"]["items"] == []:
        return None

    results = results["tracks"]["items"][0]
    track_id = results["id"]
    audio_features = sp.audio_features(track_id)[0]

    song_data["artist"] = [artist]
    song_data["name"] = [name]
    song_data["year"] = [int(results["album"]["release_date"].split("-")[0])]
    song_data["album_url"] = [results["album"]["images"][1]["url"]]
    song_data["explicit"] = [int(results["explicit"])]
    song_data["duration_ms"] = [results["duration_ms"]]
    song_data["popularity"] = [results["popularity"]]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[
            (spotify_data["name"] == song["name"])
            & (spotify_data["artists"] == song["artist"])
        ].iloc[0]
        song_data["artist"] = song["artist"]
        results = sp.search(
            q="artist: {} track: {}".format(song["artist"], song["name"]), limit=1
        )
        song_data["album_url"] = results["tracks"]["items"][0]["album"]["images"][1][
            "url"
        ]
        return pd.DataFrame(song_data).T

    except IndexError:
        return find_song(song["artist"], song["name"])


def recommendations(artist, name):
    cols = ["artist", "name", "album_url"] + X

    seed_song_data = get_song_data({"artist": artist, "name": name}, df_data)[cols]

    seed_song_scaled = scaler_model.transform(seed_song_data[X])

    seed_song_cluster = kmeans_model.predict(seed_song_scaled)[0]

    data_same_cluster_seed_song = df_data[
        df_data["cluster"] == seed_song_cluster
    ].drop_duplicates(["artists", "name", "duration_ms"])

    scaled_same_cluster_seed_song = scaled_features[data_same_cluster_seed_song.index]

    same_cluster_songs_distances = cdist(
        seed_song_scaled, scaled_same_cluster_seed_song, "cosine"
    )[0]

    data_same_cluster_seed_song["distance"] = same_cluster_songs_distances

    recommendation_dicts = (
        data_same_cluster_seed_song.sort_values(by="distance", ascending=False)[
            ["artists", "name", "year", "id"]
        ]
        .head(10)
        .to_dict(orient="records")
    )

    for item in recommendation_dicts:
        item["album_url"] = sp.track(item["id"])["album"]["images"][0]["url"]

    return recommendation_dicts


# data and model load
df_data = pd.read_parquet("data_cleaned/data.parquet")
df_data.dropna(inplace=True)

n_clusters = 60

with open(f"model/features_raw/kmeans_{n_clusters}.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open(f"model/features_raw/scaler_1.pkl", "rb") as f:
    scaler_model = pickle.load(f)

X = [
    "valence",
    "year",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "explicit",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "popularity",
    "speechiness",
    "tempo",
]

scaled_features = scaler_model.transform(df_data[X])

cluster_predicted = kmeans_model.predict(scaled_features)
df_data["cluster"] = cluster_predicted

# spotify API authentication
scope = "user-library-read playlist-modify-private"

OAuth = SpotifyOAuth(
    scope=scope,
    redirect_uri="http://localhost:5000/callback",
    client_id="6175e5ed14114d49be4c3208d69fb95c",
    client_secret="160ae4674d5640669a908a373d1e5041",
)

client_credentials_manager = SpotifyClientCredentials(
    client_id="6175e5ed14114d49be4c3208d69fb95c",
    client_secret="160ae4674d5640669a908a373d1e5041",
)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

st.set_page_config(layout="wide")


st.write(
    """
        # I heard you like music!
        
        Want to discover similar songs to those you already love?
        
        Just put the name of the artist and song in the fields below.
        
        Based on their features like 
        - `valence`
        - `danceability`
        - `loudness`
        
        among others, we calculate the distance between musics and deliver
        to you the most similar to the one you provided.
"""
)

st.write("## Artist")
artist = st.text_input(
    label="artist",
    label_visibility="hidden",
    placeholder="Name of the artist. Ex.: Judas Priest",
)

st.write("## Song")
song = st.text_input(
    label="song", label_visibility="hidden", placeholder="Name of the song. Ex.: Angel"
)


if artist == "" or song == "":
    search_button_disable = True
else:
    search_button_disable = False

if st.button("Lets go!", disabled=search_button_disable):
    st.write(f"We are calculating songs mathematicaly closer to")

    seed_song_data = get_song_data({"artist": artist, "name": song}, df_data)
    seed_album_image = io.imread(seed_song_data["album_url"].iloc[0])

    dummy_cols = st.columns(3)

    with dummy_cols[0]:
        st.write("")
    with dummy_cols[1]:
        # columns for album and details of seed song
        seed_image_col, seed_given_col = st.columns(2)
        with seed_image_col:
            st.image(seed_album_image)
        with seed_given_col:
            st.write(f"# {song}")
            st.write(f"#### {artist}")
            st.write(f"#### {seed_song_data['year'].iloc[0]}")
    with dummy_cols[2]:
        st.write("")

    recommendations_dict = recommendations(artist=artist, name=song)
    # st.write(recommendations_dict)

    st.write("# Here is your recommendation list!")
    st.write("I hope you enjoy :)")
    st.write("")
    st.write("")

    cols_row_1 = st.columns(5)
    cols_row_2 = st.columns(5)

    for counter, recommended_song in enumerate(recommendations_dict):
        image = io.imread(recommended_song["album_url"])
        artist = recommended_song["artists"].split(";")[0]
        name = recommended_song["name"]
        year = recommended_song["year"]

        if counter < 5:
            with cols_row_1[counter]:
                st.write(artist)
                st.write(name)
                st.write(year)
                st.image(image)

        else:
            with cols_row_2[counter - 5]:
                st.write(artist)
                st.write(name)
                st.write(year)
                st.image(image)

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials

import os
from dotenv import load_dotenv

print("started file")
load_dotenv()

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=os.getenv('CLIENT_ID'), client_secret=os.getenv('CLIENT_SECRET')))
'''
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=os.getenv('CLIENT_ID'),
                                               redirect_uri="https://accounts.spotify.com/authorize?client_id=2738073cd5c942bdb9c4174b9606acab&response_type=code&redirect_uri=http%3A%2F%2Flocalhost%2F&scope=user-library-read",
                                               client_secret=os.getenv('CLIENT_SECRET'),
                                               scope="user-library-read"))'''

def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

rock_playlist_tracks = get_playlist_tracks('07L9Zwa5MbAHS078dO7Hy7')
main_playlist_tracks = get_playlist_tracks('4PcbZ9cmAwAkTjdN79XiT0')

def get_track_attributes(track_id):
    features = sp.audio_features(track_id)
    return features[0]

def create_album_dataset(album_id, rock_playlist_tracks, main_playlist_tracks):
    album_tracks = sp.album_tracks(album_id)['items']
    
    data = []
    for track in album_tracks:
        track_id = track['id']
        attributes = get_track_attributes(track_id)
        attributes['in_rock'] = 1 if any(t['track']['id'] == track_id for t in rock_playlist_tracks) else 0
        attributes['in_main'] = 1 if any(t['track']['id'] == track_id for t in main_playlist_tracks) else 0
        data.append(attributes)
    return pd.DataFrame(data)

def get_albums_in_playlist(tracks):
    album_ids = []
    for track in tracks:
        album_id = track['track']['album']['id']
        if album_id not in album_ids:
            album_ids.append(album_id)
    return album_ids


album_ids = get_albums_in_playlist(rock_playlist_tracks)
all_data = pd.concat([create_album_dataset(album_id, rock_playlist_tracks, main_playlist_tracks) for album_id in album_ids], ignore_index=True)


features = ['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
X = all_data[features]
y_rock = all_data['in_rock']
y_main = all_data['in_main']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_rock, test_size=0.2, random_state=42)

clf_rock = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rock.fit(X_train, y_train)

# Evaluate the model
y_pred = clf_rock.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Rock Playlist Prediction Accuracy: {accuracy}')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_main, test_size=0.2, random_state=42)

clf_main = RandomForestClassifier(n_estimators=100, random_state=42)
clf_main.fit(X_train, y_train)

# Evaluate the model
y_pred = clf_main.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Main Playlist Prediction Accuracy: {accuracy}')


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("spotify_millsongdata.csv")

# Keep required columns only
df = df[['song', 'artist', 'text']]
df.dropna(inplace=True)

# Combine features (Content-based)
df['combined'] = df['song'] + " " + df['artist'] + " " + df['text']

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(df['combined'])

# ---------------------------
# Cosine Similarity
# ---------------------------
similarity = cosine_similarity(feature_vectors)

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Song Recommendation System")

class SongRequest(BaseModel):
    song_name: str

@app.get("/")
def home():
    return {
        "message": "Song Recommendation API is running successfully"
    }

@app.post("/recommend")
def recommend_song(request: SongRequest):
    song_name = request.song_name

    list_of_songs = df['song'].tolist()
    close_match = difflib.get_close_matches(song_name, list_of_songs)

    if not close_match:
        return {"error": "Song not found in dataset"}

    index = df[df.song == close_match[0]].index[0]

    similarity_scores = list(enumerate(similarity[index]))
    sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sorted_songs[1:11]:
        song_index = i[0]
        recommendations.append({
            "song": df.iloc[song_index]['song'],
            "artist": df.iloc[song_index]['artist']
        })

    return {
        "input_song": close_match[0],
        "recommended_songs": recommendations
    }
import logging
from flask import Flask, request, jsonify
import pickle
import os
from datetime import datetime
from collections import Counter

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get('MODEL_PATH', '/app/data/rules.pkl')
model = None
model_last_modified = None


def load_model():
    global model, model_last_modified
    logging.info(f"Attempting to load model from {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        current_modified = os.path.getmtime(MODEL_PATH)
        if model is None or current_modified != model_last_modified:
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                model_last_modified = current_modified
                logging.info(f"Model loaded successfully. Number of songs: {len(model)}")
                logging.info(f"Sample of model content: {list(model.items())[:2]}")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}", exc_info=True)
                model = None
    else:
        logging.error(f"Model file not found at {MODEL_PATH}")

@app.route('/')
def home():
    return "Playlist Recommender API is running!"


@app.route('/api/recommend', methods=['POST'])
def recommend():
    if model is None:
        load_model()
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    songs = request.json.get('songs', [])
    if not songs:
        return jsonify({"error": "No songs provided"}), 400

    logging.info(f"Received request for songs: {songs}")

    recommendations = Counter()
    for song in songs:
        if song in model:
            recommendations.update(model[song])

    # Remove input songs from recommendations
    for song in songs:
        if song in recommendations:
            del recommendations[song]

    top_recommendations = [song for song, score in recommendations.most_common(5)]

    return jsonify({
        "recommendations": top_recommendations,
        "model_version": "1.4",
        "model_date": datetime.fromtimestamp(model_last_modified).isoformat() if model_last_modified else None,
        "num_rules": len(model),
        "num_unique_songs": len(set(song for songs in model.values() for song in songs))
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    if model is None:
        load_model()

    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_date": datetime.fromtimestamp(model_last_modified).isoformat() if model_last_modified else None,
        "num_rules": model['num_rules'] if model else None,
        "num_unique_songs": model['num_unique_songs'] if model else None
    })


if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=7000)
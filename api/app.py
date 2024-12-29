from collections import Counter
from flask import Flask, request, jsonify
import pickle
import os
from datetime import datetime
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get('MODEL_PATH', '../data/rules.pkl')
model = None
model_info = None
model_last_modified = None

def load_model():
    global model, model_info, model_last_modified
    if os.path.exists(MODEL_PATH):
        current_modified = os.path.getmtime(MODEL_PATH)
        if model is None or current_modified != model_last_modified:
            try:
                with open(MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                    model = data['rules']
                    model_info = data['info']
                model_last_modified = current_modified
                logging.info(f"Model loaded successfully. Last modified: {datetime.fromtimestamp(model_last_modified)}")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                model = model_info = None
    else:
        logging.error(f"Model file not found at {MODEL_PATH}")

def get_frequent_songs(n=10):
    if model is None:
        return []
    all_songs = [song for sublist in model['antecedents'] for song in sublist] + \
                [song for sublist in model['consequents'] for song in sublist]
    return [song for song, _ in Counter(all_songs).most_common(n)]

@app.route('/api/recommend', methods=['POST'])
def recommend():
    load_model()
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    songs = set(request.json.get('songs', []))
    if not songs:
        return jsonify({"error": "No songs provided"}), 400

    logging.info(f"Received request for songs: {songs}")

    recommendations = set()
    for song in songs:
        relevant_rules = model[model['antecedents'].apply(lambda x: song in x)]
        for _, rule in relevant_rules.iterrows():
            recommendations.update(rule['consequents'])

    recommendations = list(recommendations - songs)

    if not recommendations:
        recommendations = [song for song in get_frequent_songs() if song not in songs]

    return jsonify({
        "recommendations": recommendations[:5],
        "model_version": datetime.fromtimestamp(model_last_modified).isoformat() if model_last_modified else None,
    })

# @app.route('/api/song_info', methods=['GET'])
# def song_info():
#     load_model()
#     if model is None:
#         return jsonify({"error": "Model not available"}), 500
#
#     song_name = request.args.get('song', '')
#     if not song_name:
#         return jsonify({"error": "No song name provided"}), 400
#
#     relevant_rules = model[model['antecedents'].apply(lambda x: song_name in x)]
#     as_consequent = model[model['consequents'].apply(lambda x: song_name in x)]
#
#     return jsonify({
#         "song": song_name,
#         "in_antecedents": len(relevant_rules),
#         "in_consequents": len(as_consequent),
#         "sample_antecedent_rules": relevant_rules.head(5).to_dict(orient='records') if not relevant_rules.empty else [],
#         "sample_consequent_rules": as_consequent.head(5).to_dict(orient='records') if not as_consequent.empty else []
#     })

# @app.route('/api/frequent_songs', methods=['GET'])
# def frequent_songs():
#     n = request.args.get('n', default=10, type=int)
#     return jsonify({"frequent_songs": get_frequent_songs(n)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10100)
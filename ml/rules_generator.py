import pandas as pd
from collections import Counter
import pickle
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_dataset(dataset_path, chunksize=100000):
    playlist_tracks = {}
    all_tracks = set()
    for chunk in pd.read_csv(dataset_path, chunksize=chunksize, usecols=['pid', 'track_name']):
        grouped = chunk.groupby('pid')['track_name'].apply(set).to_dict()
        for pid, tracks in grouped.items():
            if pid not in playlist_tracks:
                playlist_tracks[pid] = set()
            playlist_tracks[pid].update(tracks)
            all_tracks.update(tracks)
    return playlist_tracks, all_tracks

def generate_simple_rules_in_stream(playlist_tracks, min_support=10, max_rules=1000000):
    track_counts = Counter(track for tracks in playlist_tracks.values() for track in tracks)
    frequent_tracks = {track for track, count in track_counts.items() if count >= min_support}
    rules_generated = 0
    for pid, tracks in playlist_tracks.items():
        frequent_tracks_in_playlist = frequent_tracks.intersection(tracks)
        for track in frequent_tracks_in_playlist:
            for other_track in frequent_tracks_in_playlist - {track}:
                yield (frozenset([track]), frozenset([other_track]),
                       track_counts[other_track] / len(playlist_tracks))
                rules_generated += 1
                if rules_generated >= max_rules:
                    return

def generate_model(dataset_paths, output_path, songs_dataset_path=None, chunksize=100000):
    all_playlist_tracks, all_tracks = {}, set()
    for dataset_path in dataset_paths:
        playlist_tracks, tracks = load_and_process_dataset(dataset_path, chunksize)
        all_playlist_tracks.update(playlist_tracks)
        all_tracks.update(tracks)
    if songs_dataset_path:
        all_tracks.update(pd.read_csv(songs_dataset_path, usecols=['track_name'])['track_name'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_file = output_path + ".tmp"
    with open(temp_file, 'w') as f:
        for rule in generate_simple_rules_in_stream(all_playlist_tracks):
            f.write(f"{rule}\n")
    antecedent_counts, consequent_counts = Counter(), Counter()
    with open(temp_file, 'r') as f:
        for line in f:
            rule = eval(line.strip())
            antecedent_counts.update(rule[0])
            consequent_counts.update(rule[1])
    model_info = {
        'num_rules': sum(1 for _ in open(temp_file)),
        'top_antecedents': antecedent_counts.most_common(20),
        'top_consequents': consequent_counts.most_common(20),
        'num_unique_songs': len(all_tracks)
    }
    with open(output_path, 'wb') as f:
        pickle.dump({'rules_file': temp_file, 'info': model_info}, f)

if __name__ == "__main__":
    dataset_paths = [
        '/app/datasets/2023_spotify_ds1.csv',
        '/app/datasets/2023_spotify_ds2.csv'
    ]
    songs_dataset_path = '/app/datasets/2023_spotify_songs.csv'
    output_path = '/app/data/rules.pkl'
    os.makedirs('/app/data', exist_ok=True)

    generate_model(dataset_paths, output_path, songs_dataset_path)
    print(f"Modelo gerado e salvo em {output_path}")
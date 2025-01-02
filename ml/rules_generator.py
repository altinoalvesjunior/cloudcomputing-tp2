import pandas as pd
from collections import defaultdict
import pickle
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_process_dataset(dataset_path, chunksize=100000):
    playlist_tracks = defaultdict(set)
    for chunk in pd.read_csv(dataset_path, chunksize=chunksize, usecols=['pid', 'track_name', 'artist_name']):
        chunk['song'] = chunk['artist_name'] + ' - ' + chunk['track_name']
        for _, row in chunk.iterrows():
            playlist_tracks[row['pid']].add(row['song'])
    return playlist_tracks


def generate_rules(playlist_tracks, min_support=10, min_confidence=0.5):
    item_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    total_playlists = len(playlist_tracks)

    for tracks in tqdm(playlist_tracks.values(), desc="Counting items and pairs"):
        for track in tracks:
            item_counts[track] += 1
        for i, track1 in enumerate(tracks):
            for track2 in list(tracks)[i + 1:]:
                if track1 < track2:
                    pair_counts[(track1, track2)] += 1
                else:
                    pair_counts[(track2, track1)] += 1

    rules = defaultdict(dict)
    for (track1, track2), pair_count in tqdm(pair_counts.items(), desc="Generating rules"):
        if pair_count >= min_support:
            confidence1 = pair_count / item_counts[track1]
            confidence2 = pair_count / item_counts[track2]
            if confidence1 >= min_confidence:
                rules[track1][track2] = confidence1
            if confidence2 >= min_confidence:
                rules[track2][track1] = confidence2

    return dict(rules)


def generate_model(dataset_paths, output_path, songs_dataset_path=None):
    all_playlist_tracks = defaultdict(set)
    all_tracks = set()

    for dataset_path in dataset_paths:
        playlist_tracks = load_and_process_dataset(dataset_path)
        for pid, tracks in playlist_tracks.items():
            all_playlist_tracks[pid].update(tracks)
        all_tracks.update([track for tracks in playlist_tracks.values() for track in tracks])

    if songs_dataset_path:
        songs_df = pd.read_csv(songs_dataset_path)
        songs_df['song'] = songs_df['artist_name'] + ' - ' + songs_df['track_name']
        all_tracks.update(songs_df['song'])

    rules = generate_rules(all_playlist_tracks)

    model = {
        'rules': rules,
        'num_rules': sum(len(r) for r in rules.values()),
        'num_unique_songs': len(all_tracks)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

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
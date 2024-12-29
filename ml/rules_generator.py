import pandas as pd
from collections import Counter
import pickle
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_dataset(dataset_path):
    logging.info(f"Processando dataset: {dataset_path}")

    chunks = pd.read_csv(dataset_path, chunksize=100000)
    playlist_tracks = {}
    all_tracks = set()

    for chunk in tqdm(chunks, desc="Processando chunks"):
        for _, row in chunk.iterrows():
            pid = row['pid']
            track = row['track_name']
            if pid not in playlist_tracks:
                playlist_tracks[pid] = set()
            playlist_tracks[pid].add(track)
            all_tracks.add(track)

    return playlist_tracks, all_tracks


def generate_simple_rules(playlist_tracks, min_support=10, max_rules=1000000):
    logging.info("Gerando regras simples...")
    track_counts = Counter([track for tracks in playlist_tracks.values() for track in tracks])
    frequent_tracks = {track for track, count in track_counts.items() if count >= min_support}

    rules = []
    for pid, tracks in tqdm(playlist_tracks.items(), desc="Gerando regras"):
        frequent_tracks_in_playlist = frequent_tracks.intersection(tracks)
        for track in frequent_tracks_in_playlist:
            other_tracks = frequent_tracks_in_playlist - {track}
            for other_track in other_tracks:
                rules.append(
                    (frozenset([track]), frozenset([other_track]), track_counts[other_track] / len(playlist_tracks)))
                if len(rules) >= max_rules:
                    return rules
    return rules


def generate_model(dataset_paths, output_path, songs_dataset_path=None):
    logging.info("Iniciando a geração do modelo")

    all_playlist_tracks = {}
    all_tracks = set()

    for dataset_path in dataset_paths:
        playlist_tracks, tracks = load_and_process_dataset(dataset_path)
        all_playlist_tracks.update(playlist_tracks)
        all_tracks.update(tracks)

    if songs_dataset_path:
        songs_df = pd.read_csv(songs_dataset_path)
        all_tracks.update(songs_df['track_name'])

    logging.info(f"Total de playlists: {len(all_playlist_tracks)}")
    logging.info(f"Total de músicas únicas: {len(all_tracks)}")

    rules = generate_simple_rules(all_playlist_tracks)

    logging.info(f"Total de regras geradas: {len(rules)}")

    rules_df = pd.DataFrame(rules, columns=['antecedents', 'consequents', 'confidence'])
    rules_df['lift'] = rules_df['confidence'] / (
        rules_df['consequents'].apply(lambda x: len(x) / len(all_playlist_tracks)))
    rules_df = rules_df.sort_values('lift', ascending=False).head(1000000)

    antecedent_counts = Counter([item for items in rules_df['antecedents'] for item in items])
    consequent_counts = Counter([item for items in rules_df['consequents'] for item in items])

    model_info = {
        'num_rules': len(rules_df),
        'top_antecedents': antecedent_counts.most_common(20),
        'top_consequents': consequent_counts.most_common(20),
        'num_unique_songs': len(all_tracks)
    }

    logging.info(f"Salvando modelo em {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump({'rules': rules_df, 'info': model_info}, f)

    logging.info("Geração do modelo completa")
    logging.info(f"Informações do modelo: {model_info}")

if __name__ == "__main__":
    dataset_paths = [
        '../2023_spotify_ds1.csv',
        '../2023_spotify_ds2.csv'
    ]
    songs_dataset_path = '../2023_spotify_songs.csv'
    output_path = '../data/rules.pkl'

    generate_model(dataset_paths, output_path, songs_dataset_path)
    print(f"Modelo gerado e salvo em {output_path}")
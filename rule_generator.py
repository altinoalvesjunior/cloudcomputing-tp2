from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import pickle

# Carrega os dados de playlists
playlists = pd.read_csv('2023_spotify_ds1.csv', header=0)

# Agrupa as músicas de cada playlist, mantendo apenas as músicas
playlist_items = playlists.groupby('pid')['track_name'].agg(list).reset_index()

# Cria uma lista de listas, onde cada lista interna representa um 'cesto' de músicas
baskets = playlist_items['track_name'].values.tolist()

# Converte a lista de listas para o formato "one-hot encoded"
# Cria um DataFrame onde cada coluna é uma música e as linhas representam cestas (playlists)
all_songs = set(song for basket in baskets for song in basket)  # Conjunto com todas as músicas
baskets_df = pd.DataFrame([{song: (song in basket) for song in all_songs} for basket in baskets])

# Executa o algoritmo Apriori para encontrar conjuntos de músicas frequentes
frequent_itemsets = apriori(baskets_df, min_support=0.05, use_colnames=True)

# Calcula o número de itemsets
num_itemsets = len(frequent_itemsets)

# Gera as regras de associação
rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric='confidence', min_threshold=0.7)

# Converte os frozensets para listas para facilitar a serialização
rules['antecedents'] = rules['antecedents'].apply(list)
rules['consequents'] = rules['consequents'].apply(list)

# Salva as regras em um arquivo pickle
with open('rules.pkl', 'wb') as f:
    pickle.dump(rules, f)
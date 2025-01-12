import requests
import time
from datetime import datetime
import random

PORT = 30502
url = f"http://localhost:{PORT}/api/recommend"

all_songs = [
    "Ariana Grande - 7 rings",
    "Lady Gaga - Shallow",
    "Justin Bieber - Sorry",
    "Camila Cabello - Havana",
    "Lil Nas X - Old Town Road",
    "Shawn Mendes - Señorita",
    "Imagine Dragons - Believer",
    "Bruno Mars - Uptown Funk",
    "Adele - Hello",
    "Coldplay - Viva La Vida",
    "Katy Perry - Roar",
    "Rihanna - Work"
]

def get_random_songs(n=5):
    return random.sample(all_songs, n)

def log_response(response, start_time, input_songs):
    end_time = time.time()
    latency = end_time - start_time
    timestamp = datetime.now().isoformat()
    try:
        json_response = response.json()
        version = json_response.get('model_version', 'N/A')
        dataset_date = json_response.get('model_date', 'N/A')
        recommendations = json_response.get('recommendations', [])
        num_rules = json_response.get('num_rules', 'N/A')
        num_unique_songs = json_response.get('num_unique_songs', 'N/A')
        print(f"{timestamp} - Status: {response.status_code}, Version: {version}, Dataset Date: {dataset_date}, Latency: {latency:.3f}s")
        print(f"Input Songs: {input_songs}")
        print(f"Recommendations: {recommendations}")
        print(f"Number of Rules: {num_rules}, Number of Unique Songs: {num_unique_songs}")
    except ValueError:
        print(f"{timestamp} - Status: {response.status_code}, Não foi possível decodificar JSON, Latency: {latency:.3f}s")
    print("-" * 80)  # Linha separadora para melhor legibilidade

print(f"Iniciando teste de recomendação. Enviando requisições para {url}")

while True:
    try:
        input_songs = get_random_songs()
        data = {"songs": input_songs}
        start_time = time.time()
        response = requests.post(url, json=data, timeout=5)
        log_response(response, start_time, input_songs)
    except requests.exceptions.RequestException as e:
        print(f"{datetime.now().isoformat()} - Error: {e}")
    time.sleep(1)
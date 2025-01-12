import requests
import json
import sys

NODE_IP = "192.168.121.48"
PORT = 30502

BASE_URL = f"http://{NODE_IP}:{PORT}"


def test_root():
    response = requests.get(BASE_URL)
    print("Root endpoint test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    print()


def test_recommend(songs):
    url = f"{BASE_URL}/api/recommend"
    payload = {"songs": songs}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print("Recommend endpoint test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json() if response.status_code == 200 else response.text}")
    print()


if __name__ == "__main__":
    test_root()

    if len(sys.argv) > 1:
        songs = sys.argv[1:]
        test_recommend(songs)
    else:
        test_cases = [
            ["Ed Sheeran - Shape of You"],
            ["The Weeknd - Blinding Lights", "Dua Lipa - Don't Start Now"],
            ["Taylor Swift - Shake It Off", "Maroon 5 - Sugar", "Justin Bieber - Sorry"]
        ]
        for case in test_cases:
            test_recommend(case)

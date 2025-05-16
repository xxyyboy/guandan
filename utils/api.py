# 2025/5/16 12:57
import requests

API_BASE = "https://guandan-api.onrender.com"

def join_room(room_id, name, seat, model):
    res = requests.post(f"{API_BASE}/join_room", params={
        "room_id": room_id,
        "player_name": name,
        "seat": seat,
        "model": model
    })
    return res.json() if res.status_code == 200 else None

def get_room_state(room_id):
    res = requests.get(f"{API_BASE}/room_state/{room_id}")
    return res.json() if res.status_code == 200 else None

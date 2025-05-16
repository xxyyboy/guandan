# 2025/5/12 17:33
import os

project_structure = {
    "server": {
        "__init__.py": "",
        "main.py": "",  # will be filled with FastAPI app
        "schemas.py": "",  # Pydantic models
        "state.py": "",  # in-memory game state
    }
}

# 文件内容（稍后填充）
main_py = """\
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from server.schemas import PlayerConfig, RoomState
from server.state import room_store

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/join_room")
def join_room(room_id: str, player_name: str, seat: int, model: str):
    return room_store.join_room(room_id, player_name, seat, model)

@app.post("/leave_room")
def leave_room(room_id: str, seat: int):
    return room_store.leave_seat(room_id, seat)

@app.post("/start_game")
def start_game(room_id: str):
    return room_store.start_game(room_id)

@app.get("/room_state/{room_id}")
def get_room_state(room_id: str):
    return room_store.get_state(room_id)
"""

schemas_py = """\
from pydantic import BaseModel
from typing import Optional, List, Dict

class PlayerConfig(BaseModel):
    name: str
    seat: int
    model: str

class RoomState(BaseModel):
    room_id: str
    players: Dict[int, PlayerConfig]
    host: Optional[int]
    game_started: bool
"""

state_py = """\
from typing import Dict, Optional
from server.schemas import RoomState, PlayerConfig

class RoomStore:
    def __init__(self):
        self.rooms: Dict[str, RoomState] = {}

    def join_room(self, room_id: str, player_name: str, seat: int, model: str):
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomState(room_id=room_id, players={}, host=seat, game_started=False)
        room = self.rooms[room_id]
        if room.game_started:
            raise Exception("Game already started")
        if seat in room.players:
            raise Exception("Seat already taken")
        room.players[seat] = PlayerConfig(name=player_name, seat=seat, model=model)
        return room

    def leave_seat(self, room_id: str, seat: int):
        if room_id not in self.rooms or seat not in self.rooms[room_id].players:
            raise Exception("Invalid leave request")
        del self.rooms[room_id].players[seat]
        return {"status": "left"}

    def start_game(self, room_id: str):
        if room_id not in self.rooms:
            raise Exception("Room not found")
        self.rooms[room_id].game_started = True
        return {"status": "started"}

    def get_state(self, room_id: str):
        if room_id not in self.rooms:
            raise Exception("Room not found")
        return self.rooms[room_id]

room_store = RoomStore()
"""

project_structure["server"]["main.py"] = main_py
project_structure["server"]["schemas.py"] = schemas_py
project_structure["server"]["state.py"] = state_py

project_structure

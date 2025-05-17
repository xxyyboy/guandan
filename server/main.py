# 2025/5/16 21:52
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from server.schemas import PlayerConfig, RoomState
from server.state import room_store

app = FastAPI()

# 允许跨域请求（前端用 Streamlit，跨域请求这个后端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 如需限制安全，可改为具体地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Guandan server is running.", "Creator": "github.com/746505972"}

@app.get("/room_state/{room_id}")
def get_room_state(room_id: str):
    return room_store.get_state(room_id)

@app.post("/join_room")
def join_room(
    room_id: str,
    player_name: str = Query(...),
    seat: int = Query(...),
    model: str = Query(...)
):
    """
    玩家或 AI 加入房间的指定座位。
    model: 可以是 "ai" 或 "user:<user_id>"
    """
    try:
        return room_store.join_room(room_id, player_name, seat, model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/leave_room")
def leave_room(room_id: str, seat: int = Query(...)):
    try:
        return room_store.leave_seat(room_id, seat)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/start_game")
def start_game(room_id: str):
    try:
        return room_store.start_game(room_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

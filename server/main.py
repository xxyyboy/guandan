from test import GuandanGame,M
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from server.schemas import PlayerConfig, RoomState
from fastapi.responses import JSONResponse
from server.state import room_store
from pydantic import BaseModel
import uuid,os
app = FastAPI()

# 添加 CORS 中间件，允许所有来源（开发环境适用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应限制为前端域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法（GET/POST/PUT 等）
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["*"]  # 允许浏览器访问自定义头
)

@app.get("/list_models", response_class=JSONResponse)
def list_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
    model_dir = os.path.join(current_dir, "..", "models")     # 拼接模型目录
    # print("📂 模型目录：", model_dir)
    try:
        models = [f for f in os.listdir(model_dir)
                  if f.endswith(".pth") and (f.startswith("a") or f.startswith("s"))]
        models.sort(key=lambda x: 0 if x == "show2.pth" else 1)
        # print("📂 可用模型列表：", models)
        return JSONResponse(content={"models": models})
    except Exception as e:
        return JSONResponse(content={"models": [], "error": str(e)})

# 保存每个用户的游戏实例（模拟 session）
solo_sessions = {}

class SoloGameConfig(BaseModel):
    model: str
    position: int
    user_id: str

@app.post("/create_solo_game")
def create_solo_game(config: SoloGameConfig):
    # print(f"接收参数: position={config.position}, type={type(config.position)}")  # 调试
    game = GuandanGame(
        user_player=int(config.position),  # 强制转换为整数
        verbose=False,
        model_path=os.path.join("models", config.model)
    )
    print(f"游戏初始化完成: user_player={game.user_player}")  # 验证
    solo_sessions[config.user_id] = game
    return {"status": "solo game created", "user_id": config.user_id}



@app.get("/solo_state/{user_id}")
def solo_state(user_id: str):
    game = solo_sessions.get(user_id)
    if not game:
        return {"error": "未找到游戏实例"}

    state = game.get_game_state()

    return {
        "hand": [game.players[i].hand for i in range(4)], # 手牌
        "user_hand": game.players[game.user_player].hand, # 用户手牌
        "hand_size": [len(game.players[i].hand) for i in range(4)], # 手牌数量
        "last_play": game.last_play, # 上一次有效出牌
        "last_player": game.last_player, # 上一次出牌的玩家
        "last_plays": [game.players[i].last_played_cards for i in range(4)], # 所有人上次出牌
        "current_player": game.current_player,
        "user_player": game.user_player,
        "history": state["history"],
        "ranking": game.ranking,
        "is_game_over": game.is_game_over,
        "is_free_turn": game.is_free_turn,
        "pass_count": game.pass_count,
        "last_play_type": game.map_cards_to_action(game.last_play, M, game.active_level)["type"] if game.last_play else "无",
        "ai_suggestions": game.get_ai_suggestions(),
        "active_level": game.point_to_card(game.active_level),
        
    }
    
@app.post("/solo_play_card")
def solo_play_card(data: dict):
    user_id = data["user_id"]
    cards = data["cards"]
    game = solo_sessions.get(user_id)
    if not game:
        return {"error": "无此游戏"}
    return game.submit_user_move(cards)

@app.post("/solo_autoplay")
def solo_autoplay(data: dict):
    user_id = data["user_id"]
    game = solo_sessions.get(user_id)
    if not game:
        return {"error": "无此游戏"}
    game.step()  # 循环执行一步
    return {"status": "autoplay step executed"}

@app.post("/solo_new_game")
def solo_new_game(data: dict):
    user_id = data["user_id"]
    model = data["model"]
    position = data["position"]
    game = GuandanGame(
        user_player=int(position),  # 强制转换为整数
        verbose=False,
        model_path=os.path.join("models", model)
    )
    solo_sessions[user_id] = game
    return {"status": "new game created"}

    
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

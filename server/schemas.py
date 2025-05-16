# 2025/5/16 21:55
from pydantic import BaseModel
from typing import Dict, Optional

class PlayerConfig(BaseModel):
    name: str             # 显示的昵称（AI 可为 "AI"）
    seat: int             # 座位编号（0~3）
    model: str            # "ai" 或 "user:<user_id>"
    id: Optional[str] = None  # 用户唯一 ID，仅对 user 类型有值

class RoomState(BaseModel):
    room_id: str
    players: Dict[int, PlayerConfig]  # seat -> PlayerConfig
    host: Optional[int]               # 房主座位号
    game_started: bool

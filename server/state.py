# 2025/5/16 21:56
from typing import Dict
from server.schemas import RoomState, PlayerConfig

class RoomStore:
    def __init__(self):
        self.rooms: Dict[str, RoomState] = {}  # room_id -> RoomState

    def get_or_create_room(self, room_id: str) -> RoomState:
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomState(
                room_id=room_id,
                players={},
                host=None,
                game_started=False
            )
        return self.rooms[room_id]

    def join_room(self, room_id: str, player_name: str, seat: int, model: str):
        room = self.get_or_create_room(room_id)

        if room.game_started:
            raise Exception("游戏已开始，无法加入")

        if seat in room.players:
            raise Exception(f"座位 {seat} 已被占用")

        if model.startswith("user:"):
            user_id = model.split("user:")[1]
        else:
            user_id = None  # AI 不需要 ID

        player = PlayerConfig(
            name=player_name,
            seat=seat,
            model=model,
            id=user_id
        )

        room.players[seat] = player

        # 设置房主
        if room.host is None:
            room.host = seat

        return room

    def leave_seat(self, room_id: str, seat: int):
        if room_id not in self.rooms:
            raise Exception("房间不存在")

        room = self.rooms[room_id]
        if seat not in room.players:
            raise Exception("该座位未被占用")

        del room.players[seat]

        # 如果房主离开，则自动指定下一个加入的玩家为房主
        if room.host == seat:
            room.host = next(iter(room.players), None)

        return {"status": "left"}

    def start_game(self, room_id: str):
        if room_id not in self.rooms:
            raise Exception("房间不存在")

        room = self.rooms[room_id]
        if room.game_started:
            raise Exception("游戏已开始")

        # 填补 AI 玩家
        for seat in range(4):
            if seat not in room.players:
                room.players[seat] = PlayerConfig(
                    name=f"AI-{seat}",
                    seat=seat,
                    model="ai"
                )

        room.game_started = True
        return {"status": "started", "players": room.players}

    def get_state(self, room_id: str):
        if room_id not in self.rooms:
            raise Exception("房间不存在")
        return self.rooms[room_id]

room_store = RoomStore()

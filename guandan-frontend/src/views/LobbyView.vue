<template>
  <div>
    <h1>🕹️ 掼蛋联机大厅</h1>

    <div>
      <!-- 房间号输入框 -->
      <input
        type="text"
        v-model="roomId"
        placeholder="请输入房间号"
        class="input-room"
      />
      <p>房间号：{{ roomId }}</p>
    </div>

    <div class="player">
      <!-- 玩家座位 -->
      <div v-for="(player, index) in players" :key="index" class="player">
        <div v-if="player">
          <strong>玩家 {{ index + 1 }}</strong>
          <div v-if="player.name">
            <p>{{ player.name }}</p>
            <p>模型：{{ player.model || "AI" }}</p>
            <button @click="leaveSeat(index)">离开</button>
          </div>
          <div v-else>
            <input v-model="playerNames[index]" placeholder="请输入名字" />
            <button @click="joinSeat(index)">加入</button>
          </div>
        </div>
      </div>
    </div>

    <div v-if="joinedIndex === hostSeat">
      <button @click="startGame" :disabled="!canStartGame">🚀 开始游戏</button>
    </div>

    <button @click="leaveRoom">🔙 离开房间</button>
  </div>
</template>

<script>
import { ref, onMounted } from "vue";
import axios from "axios";

export default {
  data() {
    return {
      roomId: "", // 房间号
      players: [null, null, null, null], // 玩家座位
      playerNames: ["", "", "", ""], // 玩家名字输入
      hostSeat: 0, // 房主座位（默认 1号座位）
      joinedIndex: null, // 当前加入的座位
      canStartGame: false, // 是否能启动游戏
    };
  },
  computed: {
    // 判断是否为房主（1号座位）
    isHost() {
      return this.joinedIndex === this.hostSeat;
    },
  },
  methods: {
    // 初始化房间状态
    async fetchRoomState() {
      try {
        const res = await axios.get(`https://precious-ideally-ostrich.ngrok-free.app/room_state/${this.roomId}`);
        this.players = res.data.players;
        this.hostSeat = res.data.host || 0;
        this.canStartGame = res.data.players.every(player => player !== null);
      } catch (error) {
        console.error("获取房间状态失败", error);
      }
    },

    // 加入座位
    async joinSeat(index) {
      try {
        const playerName = this.playerNames[index];
        const model = this.joinedIndex === null ? "user" : "ai"; // 如果已加入，自动使用AI模型
        const res = await axios.post("https://precious-ideally-ostrich.ngrok-free.app/join_room", {
          room_id: this.roomId,
          seat: index,
          player_name: playerName || `玩家 ${index + 1}`,
          model: model,
        });
        this.joinedIndex = index; // 更新已加入座位
        this.fetchRoomState(); // 更新房间状态
      } catch (error) {
        console.error("加入座位失败", error);
      }
    },

    // 离开座位
    async leaveSeat(index) {
      try {
        const res = await axios.post("https://precious-ideally-ostrich.ngrok-free.app/leave_room", {
          room_id: this.roomId,
          seat: index,
        });
        this.joinedIndex = null; // 重置已加入座位
        this.fetchRoomState(); // 更新房间状态
      } catch (error) {
        console.error("离开座位失败", error);
      }
    },

    // 启动游戏
    async startGame() {
      try {
        const res = await axios.post("https://precious-ideally-ostrich.ngrok-free.app/start_game", {
          room_id: this.roomId,
        });
        if (res.status === 200) {
          this.$router.push({ name: "game" }); // 跳转到游戏页面
        }
      } catch (error) {
        console.error("启动游戏失败", error);
      }
    },

    // 离开房间
    async leaveRoom() {
      if (this.joinedIndex !== null) {
        await this.leaveSeat(this.joinedIndex);
      }
      this.$router.push({ name: "setup" }); // 返回设置页面
    },
  },

  // 页面加载时获取房间状态
  onMounted() {
    this.roomId = this.$route.params.roomId || "room-001"; // 通过路由获取房间号（可选）
    this.fetchRoomState();
  },
};
</script>

<style scoped>
.player {
  margin-bottom: 10px;
}

.input-room {
  padding: 5px;
  font-size: 16px;
}

button {
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:disabled {
  background-color: gray;
}

button:hover {
  background-color: #0056b3;
}
</style>

<template>
  <div class="solo">
    <h2>🧑‍💻 单人对战界面</h2>
    <div v-if="!gameData">加载中...</div>

    <div v-else>
      <div style="margin-bottom: 10px;">
        <strong>当前级牌：</strong> {{ gameData.active_level }} |
        <strong>当前轮到：</strong> 玩家 {{ gameData.current_player + 1 }}
      </div>

      <!-- 玩家状态 -->
      <div class="players" style="display: flex; gap: 10px; margin-bottom: 10px;">
        <div v-for="i in 4" :key="i" :style="{ flex: 1, backgroundColor: gameData.last_player === i-1 ? '#ffe9b3' : '#f3f3f3', padding: '10px', borderRadius: '6px' }">
          <strong>玩家 {{ i }}{{ i-1 === gameData.user_player ? ' 🧑‍💻' : '' }}</strong><br />
          <div>手牌：<span :style="{ color: getHandColor(i-1) }">{{ getHandSize(i-1) }}</span> 张</div>
          <div>出牌：{{ getLastPlay(i-1) }}</div>
        </div>
      </div>

      <!-- AI建议与上次出牌 -->
      <div style="background:#eef3fa; padding:12px; border-radius:8px; display:flex; gap:20px;">
        <div style="flex: 3;">
          <strong>🤖 AI建议：</strong>
          <ul>
            <li v-for="(sug, i) in gameData.ai_suggestions" :key="i">{{ sug }}</li>
          </ul>
        </div>
        <div style="flex:1;">
          <strong>📦 上次出牌：</strong><br />
          类型：{{ gameData.last_play_type }}<br />
          内容：{{ gameData.last_play.join(' ') || '无' }}
        </div>
      </div>

      <!-- 选择出牌 -->
      <div style="margin-top: 20px;">
        <h3>🃏 你的手牌：</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          <button
            v-for="(card, index) in gameData.hand"
            :key="index"
            @click="toggleSelect(index)"
            :style="{
              padding: '8px 12px',
              borderRadius: '6px',
              border: selected.includes(index) ? '2px solid green' : '1px solid #ccc',
              background: selected.includes(index) ? '#d0f0d0' : '#fff'
            }"
          >
            {{ card }}
          </button>
        </div>

        <div style="margin-top: 10px;">
          已选：{{ selectedCards.join('、') || '无' }}
        </div>

        <div style="margin-top: 10px;">
          <button @click="submitMove">✔️ 出牌</button>
          <button @click="pass" :disabled="gameData.is_free_turn">👟 跳过</button>
          <button @click="autoPlay">🤖 自动</button>
          <button @click="refreshState">🔁 刷新</button>
        </div>

        <div v-if="gameData.is_game_over" style="margin-top: 20px;">
          <h3>🎉 游戏结束</h3>
          <p>排名：{{ gameData.ranking.map(i => '玩家 ' + (i+1)).join(' > ') }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref, computed } from 'vue'
import { useGlobalStore } from '../stores'
import { api } from '../utils/axios'

const store = useGlobalStore()
const gameData = ref<any>(null)
const selected = ref<number[]>([])

const refreshState = async () => {
  const res = await api.get(`/solo_state/${store.userId}`)
  gameData.value = res.data
}

const getHandColor = (i: number) =>
  gameData.value.ranking.includes(i) ? 'green' : gameData.value.hand_size < 3 ? 'red' : 'black'

const getHandSize = (i: number) =>
  gameData.value.user_player === i ? gameData.value.hand.length : gameData.value?.statuses?.[i]?.hand_size || '??'

const getLastPlay = (i: number) =>
  gameData.value.statuses?.[i]?.last_play?.join(' ') || 'Pass'

const toggleSelect = (idx: number) => {
  if (selected.value.includes(idx)) {
    selected.value = selected.value.filter(i => i !== idx)
  } else {
    selected.value.push(idx)
  }
}

const selectedCards = computed(() => selected.value.map(i => gameData.value.hand[i]))

const submitMove = async () => {
  const res = await api.post('/solo_play_card', {
    user_id: store.userId,
    cards: selectedCards.value
  })
  selected.value = []
  refreshState()
}

const pass = async () => {
  const res = await api.post('/solo_play_card', {
    user_id: store.userId,
    cards: []
  })
  selected.value = []
  refreshState()
}

const autoPlay = async () => {
  await api.post('/solo_autoplay', { user_id: store.userId })
  refreshState()
}

onMounted(refreshState)
</script>

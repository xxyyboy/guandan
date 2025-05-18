<template>
  <div class="solo">
    <div class="header" style="width: 100%; text-align: center; margin-bottom: 1.5rem;">
      <h2>🤖 AI 掼蛋对战演示</h2>
    </div>

    <div v-if="!gameData">加载中...</div>
    
    <div v-else class="game-container">
      <!-- 两列布局 - 主区域和侧边栏 -->
      <div class="main-content">
        <!-- 玩家状态 -->
        <div class="player-status-container">
          <div v-for="i in 4" :key="i" class="player-card" 
               :style="{ backgroundColor: i-1 === gameData.last_player ? '#ffe9b3' : '#f5f5f5' }">
            <div class="player-name">
              玩家 {{ i }}{{ i-1 === gameData.user_player ? ' 🧑‍💻' : '' }}
              <span v-if="gameData.ranking.includes(i-1)" class="player-rank">
                {{ getRankText(gameData.ranking.indexOf(i-1)) }}
              </span>
            </div>
            <div class="player-info">
              <div :style="{ color: getHandColor(i-1) }">手牌：{{ getHandSize(i-1) }} 张</div>
              <div>出牌：{{ getLastPlay(i-1) }}</div>
            </div>
          </div>
        </div>

        <!-- AI建议与上次出牌 -->
        <div class="ai-suggestion-container">
          <div class="suggestion-section">
            <h3>🤖 AI建议：</h3>
            <ul>
              <li v-for="(sug, i) in gameData.ai_suggestions" :key="i">{{ sug }}</li>
            </ul>
          </div>
          <div class="last-play-section">
            <h3>📦 上次出牌</h3>
            <div>类型：<strong>{{ gameData.last_play_type }}</strong></div>
            <div class="last-play-cards">
              {{ gameData.last_play.join(' ') || '无' }}
            </div>
          </div>
        </div>

        <!-- 游戏结束显示 -->
        <div v-if="gameData.is_game_over" class="game-over-container">
          <h3>🎉 游戏结束！</h3>
          <p><strong>最终排名：</strong></p>
          <ul>
            <li v-for="(p, i) in gameData.ranking" :key="i">
              {{ ['头游', '二游', '三游', '末游'][i] }}：玩家 {{ p + 1 }}
            </li>
          </ul>
        </div>

        <!-- 玩家行动区域 -->
        <div v-if="!gameData.is_game_over && gameData.current_player === gameData.user_player" class="player-action-container">
          <h3>🕹️ 出牌</h3>
          
          <!-- 手牌选择 -->
          <div class="hand-cards">
            <button
              v-for="(card, index) in gameData.hand[gameData.user_player]"
              :key="index"
              @click="toggleSelect(index)"
              :class="{ 'selected-card': selected.includes(index) }"
            >
              {{ convertCardDisplay(card) }}
            </button>
          </div>

          <!-- 已选牌显示 -->
          <div class="selected-cards-display">
            <strong>已选择：</strong>
            <span v-if="selectedCards.length > 0" class="has-selection">
              {{ selectedCards.map(card => convertCardDisplay(card)).join('、') }}
            </span>
            <span v-else-if="gameData.is_free_turn" class="no-selection">
              自由回合
            </span>
            <span v-else class="no-selection">
              尚未选择任何牌
            </span>
          </div>

          <!-- 操作按钮 -->
          <div class="action-buttons">
            <button @click="clearSelection" class="secondary-btn">🗑️ 清空选择</button>
            <button @click="pass" :disabled="gameData.is_free_turn" class="secondary-btn">👟 PASS</button>
            <button @click="submitMove" class="primary-btn">✔️ 确认出牌</button>
            <button @click="autoPlay" class="secondary-btn">🤖 自动</button>
          </div>
        </div>
      </div>

      <!-- 侧边栏 -->
      <div class="sidebar">
        <!-- 操作按钮 -->
        <div class="sidebar-buttons">
          <button @click="newGame" class="sidebar-btn">🔄 新一局</button>
          <button @click="goBack" class="sidebar-btn">🔙 返回设置</button>
          <a href="https://github.com/746505972/guandan" target="_blank" class="github-link">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="20">
            <span>查看项目仓库</span>
          </a>
        </div>
        <div class="badge">ver. 1.2.3</div>

        <!-- 当前状态 -->
        <div class="current-status">
          <div class="status-item">
            <strong>当前级牌：</strong>
            <span class="level-card">{{ gameData.active_level }}</span>
          </div>
          <div class="status-divider"></div>
          <div class="status-item">
            <strong>当前轮到：</strong>
            <span class="current-player">玩家 {{ gameData.current_player + 1 }}</span>
          </div>
        </div>

        <!-- 出牌历史 -->
        <div class="play-history">
          <h3>📝 出牌历史</h3>
          <textarea :value="formattedHistory" readonly></textarea>
        </div>

        <!-- 调试信息 -->
        <details class="debug-info">
          <summary>调试区</summary>
          <div class="debug-content">
            <code>is_free_turn: {{ gameData.is_free_turn }}</code>
            <code>pass_count: {{ gameData.pass_count }}</code>
            <code>jiefeng: {{ gameData.jiefeng }}</code>
            <code>{{ gameData.model_path }}</code>
            <!-- 其他调试信息可以根据需要添加 -->
          </div>
        </details>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref, computed, watch } from 'vue'
import { useGlobalStore } from '../stores'
import { api } from '../utils/axios'
import { useRouter } from 'vue-router'

const store = useGlobalStore()
const router = useRouter()
const gameData = ref<any>(null)
const selected = ref<number[]>([])
const isAutoPlaying = ref(false)

const refreshState = async () => {
  if (!store.userId) {
    console.error('userId 为空，无法获取游戏状态');
    return;
  }
  try {
    const res = await api.get(`/solo_state/${store.userId}`);
    gameData.value = res.data;
    
    // 检查是否需要自动推进游戏
    if (!gameData.value.is_game_over && 
        gameData.value.current_player !== gameData.value.user_player) {
      autoAdvanceGame();
    }
  } catch (e) {
    console.error('获取游戏状态失败', e);
  }
};

// 自动推进游戏
const autoAdvanceGame = async () => {
  if (isAutoPlaying.value) return;
  
  isAutoPlaying.value = true;
  try {
    // 持续自动推进，直到轮到玩家或游戏结束
    while (!gameData.value.is_game_over && 
           gameData.value.current_player !== gameData.value.user_player) {
      const res = await api.post('/solo_autoplay', { user_id: store.userId });
      gameData.value = res.data;
      
      // 添加短暂延迟避免过于频繁请求
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  } catch (e) {
    console.error('自动推进游戏出错', e);
  } finally {
    isAutoPlaying.value = false;
  }
};

// 监听游戏数据变化，检查是否需要自动推进
watch(() => gameData.value, (newVal) => {
  if (newVal && !newVal.is_game_over && 
      newVal.current_player !== newVal.user_player) {
    autoAdvanceGame();
  }
});

// 其他已有的方法保持不变...
const getHandColor = (i: number) => 
  gameData.value.hand_size[i] < 3 ? 'red' : 'black'

const getHandSize = (i: number) => 
  gameData.value.hand_size?.[i] || 0

const getLastPlay = (i: number) => 
  gameData.value.statuses?.[i]?.last_play?.join(' ') || 'Pass'

const getRankText = (rankIndex: number) => {
  const ranks = ["🏅头游", "🥈二游", "🥉三游", "🛑末游"];
  return ranks[rankIndex];
}

const toggleSelect = (idx: number) => {
  if (selected.value.includes(idx)) {
    selected.value = selected.value.filter(i => i !== idx)
  } else {
    selected.value.push(idx)
  }
}

const clearSelection = () => {
  selected.value = [];
}

const selectedCards = computed(() => 
  selected.value.map(i => gameData.value.hand[gameData.value.user_player][i])
)

const formattedHistory = computed(() => {
  if (!gameData.value.history) return '';
  return gameData.value.history.map((round: any[], i: number) => {
    const roundNumber = gameData.value.history.length - i;
    return `第${roundNumber}轮: ` + round.map(p => p ? p.join(' ') : 'Pass').join(' | ');
  }).join('\n');
})

const convertCardDisplay = (cardStr: string) => {
  const suitSymbols = {'黑桃': '♠️', '红桃': '♥️', '梅花': '♣️', '方块': '♦️'};
  if (cardStr === '大王') return '大王🃏';
  if (cardStr === '小王') return '小王🃟';
  for (const [cnSuit, symbol] of Object.entries(suitSymbols)) {
    if (cardStr.startsWith(cnSuit)) {
      return cardStr.replace(cnSuit, symbol);
    }
  }
  return cardStr;
}

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

const newGame = async () => {
  await api.post('/solo_new_game', { user_id: store.userId })
  refreshState()
}

const goBack = () => {
  console.log('返回设置页面');
  router.push('/')
}

onMounted(refreshState)
</script>

<style scoped>
.solo {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 1rem;
}

.game-container {
  display: flex;
  width: 100%;
  gap: 1.5rem;
}

.main-content {
  flex: 3;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.sidebar {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* 玩家状态 */
.player-status-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
}

.player-card {
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.player-name {
  font-weight: bold;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.player-rank {
  color: green;
  font-size: 0.9rem;
}

.player-info {
  font-size: 0.9rem;
}

/* AI建议区域 */
.ai-suggestion-container {
  background-color: #e3f2fd;
  border-radius: 10px;
  padding: 1.25rem;
  display: flex;
  gap: 1.25rem;
}

.suggestion-section {
  flex: 3;
}

.suggestion-section h3 {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-size: 1.1rem;
}

.suggestion-section ul {
  margin: 0;
  padding-left: 1rem;
}

.suggestion-section li {
  margin-bottom: 0.5rem;
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 6px;
}

.last-play-section {
  flex: 1;
}

.last-play-section h3 {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-size: 1.1rem;
}

.last-play-cards {
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 6px;
  margin-top: 0.5rem;
}

/* 手牌区域 */
.hand-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.hand-cards button {
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  background: white;
  cursor: pointer;
  font-size: 1.1rem;
  min-width: 3rem;
  text-align: center;
}

.hand-cards button:hover {
  transform: translateY(-2px);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.hand-cards button.selected-card {
  border: 2px solid green;
  background: #d0f0d0;
}

/* 已选牌显示 */
.selected-cards-display {
  border: 1px solid #e6e6e6;
  padding: 0.75rem;
  border-radius: 5px;
  background-color: #f9f9f9;
  margin-bottom: 1rem;
}

.has-selection {
  color: #2e7d32;
  font-weight: bold;
}

.no-selection {
  color: gray;
  font-weight: bold;
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.action-buttons button {
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  flex: 1;
  min-width: 120px;
}

.primary-btn {
  background-color: #4CAF50;
  color: white;
  border: none;
}

.secondary-btn {
  background-color: #f5f5f5;
  border: 1px solid #ccc;
}

.action-buttons button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 侧边栏样式 */
.sidebar-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.sidebar-btn {
  padding: 0.5rem;
  border-radius: 6px;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  cursor: pointer;
  font-size: 0.9rem;
  flex: 1;
  min-width: 120px;
}

.github-link {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  border-radius: 6px;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  text-decoration: none;
  color: #333;
  font-size: 0.9rem;
  flex: 1;
  min-width: 120px;
}

.badge {
  background-color: #E85889;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  width: fit-content;
}

.current-status {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  margin: 1rem 0;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.level-card {
  background-color: red;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-weight: bold;
  font-size: 1.1rem;
}

.current-player {
  color: orange;
  font-weight: bold;
  font-size: 1rem;
}

.status-divider {
  width: 1px;
  height: 1.5rem;
  background-color: #ccc;
}

.play-history textarea {
  width: 100%;
  height: 350px;
  resize: none;
  padding: 0.5rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  font-family: monospace;
  font-size: 0.9rem;
}

/* 调试信息 */
.debug-info {
  margin-top: 1rem;
}

.debug-info summary {
  font-weight: bold;
  font-size: 0.9rem;
  cursor: pointer;
}

.debug-content {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
  font-size: 0.8rem;
}

.debug-content code {
  background-color: #f5f5f5;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-family: monospace;
}

/* 游戏结束样式 */
.game-over-container {
  background-color: #e8f5e9;
  padding: 1.5rem;
  border-radius: 8px;
  margin-top: 1rem;
}

.game-over-container h3 {
  margin-top: 0;
  color: #2e7d32;
}

@media (max-width: 992px) {
  .game-container {
    flex-direction: column;
  }
  
  .player-status-container {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
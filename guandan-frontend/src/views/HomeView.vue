<template>
  <div class="home">
    <div class="header" style="text-align: center; margin-bottom: 1.5rem;">
      <h1>🎮 掼蛋设置</h1>
      <p class="subtitle">配置您的游戏参数</p>
    </div>

    <div class="settings-card" style="width: 500px; padding: 2.5rem; margin: 0 auto;">
      <div class="form-group">
        <label for="model">AI模型选择</label>
        <select 
          id="model" 
          v-model="selectedModel" 
          @change="updateModel"
          class="styled-select"
        >
          <option 
            v-for="model in availableModels" 
            :key="model" 
            :value="model"
          >
            {{ model }}
          </option>
        </select>
      </div>

      <div class="form-group">
        <label>起始位置</label>
        <select 
          v-model="selectedPosition"
          class="styled-select"
        >
          <option v-for="n in 4" :value="n - 1">
            座位 {{ n }} 
            <span v-if="n === 1">(庄家)</span>
          </option>
        </select>
      </div>

      <div class="button-group">
        <button @click="startSolo" class="primary-btn">
          <span class="icon">🎯</span> 单人对战
        </button>
        <button @click="goToLobby" class="secondary-btn">
          <span class="icon">🌐</span> 联机大厅
        </button>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useGlobalStore } from '../stores'
import { api } from '../utils/axios'
import { v4 as uuidv4 } from 'uuid'

const store = useGlobalStore()
const router = useRouter()

const availableModels = ref<string[]>([])
const selectedModel = ref<string>('')
const selectedPosition = ref(0)

onMounted(async () => {
  try {
    const res = await api.get('/list_models', {
      headers: {
        'ngrok-skip-browser-warning': 'true'  // 添加请求头跳过验证
      }
    })
    console.log('模型接口返回：', res)
    console.log('状态码:', res.status)
    console.log('响应头:', res.headers)
    console.log('响应数据类型:', typeof res.data)
    if (typeof res.data === 'string' && res.data.startsWith('<!DOCTYPE html>')) {
      console.warn('注意：返回了 HTML，接口可能有问题')
    }

    const modelList = res.data?.models || []

    if (modelList.length > 0) {
      selectedModel.value = modelList[0]
      store.setSelectedModel(modelList[0])
    } else {
      alert('❌ 模型列表为空！')
    }

    availableModels.value = modelList
    store.setModels(modelList)
  } catch (e) {
    console.error('加载模型失败', e)
    alert('无法连接后端或加载模型')
  }
})



const updateModel = () => {
  store.setSelectedModel(selectedModel.value)
}

const startSolo = async () => {
  try {
    // 确保 userId 已初始化
    store.initializeUserId();
    
    // 确保 position 是数字类型（避免字符串传递）
    const position = Number(selectedPosition.value) + 1; // 转换为 1-4 的数字
    // 检查 position 是否在有效范围内
    if (isNaN(position) || position < 1 || position > 4) {
      throw new Error('座位号必须是 1-4 的数字');
    }

    // 调试日志：打印发送的数据
    console.log('发送创建游戏请求:', {
      model: selectedModel.value,
      position: position,
      user_id: store.userId
    });

    // 发送请求
    const response = await api.post(
      '/create_solo_game',
      {
        model: selectedModel.value,
        position: position,  // 使用转换后的数字
        user_id: store.userId
      },
      {
        headers: { 
          'ngrok-skip-browser-warning': 'true',
          'Content-Type': 'application/json'  // 明确指定内容类型
        }
      }
    );

    // 调试日志：打印响应
    console.log('创建游戏响应:', response.data);

    // 存储状态并跳转
    store.setSelectedModel(selectedModel.value);
    store.setSeat(position);  // 存储数字类型的座位号
    router.push('/solo');
  } catch (e) {
    console.error('创建对局失败', e);
    alert(`创建对局失败: ${e.message}`);
  }
}

const goToLobby = () => {
  store.setSelectedModel(selectedModel.value)
  router.push('/lobby')
}


</script>

<style scoped>
.home {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  box-sizing: border-box;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
}

.header h1 {
  font-size: 2.2rem;
  color: #333;
  margin-bottom: 0.5rem;
}

.subtitle {
  font-size: 1.1rem;
  color: #666;
}

.settings-card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.form-group {
  margin-bottom: 2.2rem !important;
}

.form-group label {
  display: block;
  font-size: 1.2rem !important;
  font-weight: 500;
  margin-bottom: 1rem !important;
  color: #444;
}

.styled-select {
  width: 100%;
  padding: 0.8rem 1rem;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f9f9f9;
  transition: all 0.3s;
}

.styled-select:focus {
  outline: none;
  border-color: #646cff;
  box-shadow: 0 0 0 2px rgba(100, 108, 255, 0.2);
}

.button-group {
  display: flex;
  gap: 1rem;
  margin-top: 2.5rem !important;
}

.primary-btn, .secondary-btn {
  flex: 1;
  padding: 0.9rem;
  font-size: 1.1rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.primary-btn {
  background: #646cff;
  color: white;
}

.primary-btn:hover {
  background: #535bf2;
  transform: translateY(-2px);
}

.secondary-btn {
  background: #f0f0f0;
  color: #333;
}

.secondary-btn:hover {
  background: #e0e0e0;
  transform: translateY(-2px);
}

.icon {
  font-size: 1.2rem;
}

@media (max-width: 480px) {
  .home {
    padding: 1rem;
  }
  
  .button-group {
    flex-direction: column;
  }
}
</style>
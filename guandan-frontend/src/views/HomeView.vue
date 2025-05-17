<template>
  <div class="home">
    <h1>🎮 掼蛋设置页面</h1>

    <div style="margin-top: 20px;">
      <label for="model">请选择模型：</label>
      <select v-model="selectedModel" @change="updateModel">
        <option v-for="model in availableModels" :key="model" :value="model">
          {{ model }}
        </option>
      </select>
    </div>

    <div style="margin-top: 20px;">
      <button @click="goToSolo" style="margin-right: 20px;">🎯 单人对战</button>
      <button @click="goToLobby">🌐 联机大厅</button>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useGlobalStore } from '../stores'
import { api } from '../utils/axios'

const store = useGlobalStore()
const router = useRouter()

const availableModels = ref<string[]>([])
const selectedModel = ref<string>('')

const fetchModels = async () => {
  try {
    const res = await api.get('/list_models') // ← 你需要在后端添加该接口返回 models 文件夹下的模型名
    availableModels.value = res.data.models
    selectedModel.value = res.data.models[0] || ''
    store.setModels(res.data.models)
    store.setSelectedModel(selectedModel.value)
  } catch (e) {
    console.error('获取模型失败', e)
  }
}

const updateModel = () => {
  store.setSelectedModel(selectedModel.value)
}

const goToSolo = () => {
  store.setSelectedModel(selectedModel.value)
  router.push('/solo')
}

const goToLobby = () => {
  store.setSelectedModel(selectedModel.value)
  router.push('/lobby')
}

onMounted(fetchModels)
</script>


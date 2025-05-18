import { defineStore } from 'pinia'
import { v4 as uuidv4 } from 'uuid'

export const useGlobalStore = defineStore('global', {
  state: () => ({
    roomId: '',
    userId: '', // 全局唯一标识本客户端
    joinedSeat: null as number | null,
    availableModels: [] as string[],
    selectedModel: '',
  }),
  actions: {
    initializeUserId() {
      if (!this.userId) {
        this.userId = uuidv4()
        console.log('[store] 已生成 userId:', this.userId)
      }
    },
    setRoom(id: string) {
      this.roomId = id
    },
    setUser(id: string) {
      this.userId = id
    },
    setSeat(seat: number | null) {
      this.joinedSeat = seat
    },
    setModels(models: string[]) {
      this.availableModels = models
    },
    setSelectedModel(model: string) {
      this.selectedModel = model
    }
  }
})

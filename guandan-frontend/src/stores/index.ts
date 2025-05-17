import { defineStore } from 'pinia'

export const useGlobalStore = defineStore('global', {
  state: () => ({
    roomId: '',
    userId: '',
    joinedSeat: null as number | null,
    availableModels: [] as string[],
    selectedModel: '',
  }),
  actions: {
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

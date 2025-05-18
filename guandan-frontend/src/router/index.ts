import { createRouter, createWebHistory } from 'vue-router'
import SoloView from '../views/SoloView.vue'
import HomeView from '../views/HomeView.vue'
import LobbyView from '../views/LobbyView.vue'

const routes = [
  { path: '/', name: 'home', component: HomeView },
  { path: '/solo', name: 'solo', component: SoloView },  
  { path: '/lobby', name: 'lobby', component: LobbyView },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router

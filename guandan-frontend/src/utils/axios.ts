import axios from 'axios'

export const API_BASE = 'http://localhost:8000'
// 如果使用 ngrok，请替换为公网地址，例如：
// export const API_BASE = 'https://abc123.ngrok-free.app'

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 5000
})

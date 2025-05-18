import axios from 'axios'

export const API_BASE = 'http://localhost:8000'
// ngrok http --url=precious-ideally-ostrich.ngrok-free.app 8000
// http://localhost:8000
// https://precious-ideally-ostrich.ngrok-free.app
export const api = axios.create({
  baseURL: API_BASE,
  timeout: 5000
})
export default api
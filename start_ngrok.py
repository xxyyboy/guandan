# 2025/5/16 12:57
import subprocess

# 启动 FastAPI
uvicorn_proc = subprocess.Popen(["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"])

# 启动 ngrok（确保你已经配置过 token）
subprocess.run(["ngrok.exe", "http", "8000"])

'''
uvicorn server.main:app --host 0.0.0.0 --port 8000 
ngrok http 8000
'''

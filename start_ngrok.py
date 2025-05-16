# 2025/5/16 12:57
import subprocess

# 启动本地服务
subprocess.Popen(["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"])

# 启动 ngrok（需先安装并登录 ngrok）
subprocess.run(["ngrok", "http", "8000"])

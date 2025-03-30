# 2025/3/23 15:09
from stable_baselines3 import DQN
from guandan_env import GuandanEnv

env = GuandanEnv()
model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, batch_size=64, learning_starts=1000)
model.learn(total_timesteps=1000000)
model.save("models/guandan_dqn")

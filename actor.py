import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np
from guandan_env import GuandanGame
from get_actions import enumerate_colorful_actions  # 你已有的模块
import random
# 加载动作全集 M
with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)

# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}

# Actor 网络定义
class ActorNet(nn.Module):
    def __init__(self, state_dim=3049, action_dim=action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, mask=None):
        logits = self.net(x)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        return F.softmax(logits, dim=-1)

# 初始化模型
actor = ActorNet()
optimizer = optim.Adam(actor.parameters(), lr=1e-4)

# 训练函数
def train_on_batch(batch):
    states = torch.tensor([s["state"] for s in batch], dtype=torch.float32)
    actions = torch.tensor([s["action_id"] for s in batch], dtype=torch.long)
    rewards = torch.tensor([s["reward"] for s in batch], dtype=torch.float32)

    probs = actor(states)
    log_probs = torch.log(probs + 1e-8)
    chosen_log_probs = log_probs[range(len(batch)), actions]
    loss = -(chosen_log_probs * rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 模拟训练流程
def run_training(episodes=1000):
    for ep in range(episodes):
        game = GuandanGame()
        game.reset()
        memory = []

        while not game.check_game_over():
            player = game.players[game.current_player]
            if game.current_player == 0:
                state = game._get_obs()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(game.get_valid_action_mask(player.hand, M, game.active_level)).unsqueeze(0)
                probs = actor(state_tensor, mask)
                action_id = torch.multinomial(probs, 1).item()
                action_struct = M_id_dict[action_id]
                combos = enumerate_colorful_actions(action_struct, player.hand, game.active_level)
                move = random.choice(combos) if combos else []

                game.recent_actions[game.current_player] = move
                player.play_cards(move)
                reward = -len(player.hand)  # 越少越好
                memory.append({"state": state, "action_id": action_id, "reward": reward})
            else:
                game.ai_play(player)  # 其他人用随机

            game.current_player = (game.current_player + 1) % 4
            game.check_game_over()

        # 局结束，训练 actor
        loss = train_on_batch(memory)
        print(f"Episode {ep + 1}, loss: {loss:.4f}")

        if (ep + 1) % 100 == 0:
            torch.save(actor.state_dict(), f"actor_ep{ep+1}.pth")

if __name__ == "__main__":
    run_training()


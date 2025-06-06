"""
PPO.py
掼蛋Proximal Policy Optimization (PPO) 强化学习智能体训练代码。
实现PPO核心：策略/价值网络、GAE优势估计、损失函数裁剪、熵正则、断点续训、采样与训练流程。
"""
import os
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
from guandan_env import GuandanGame
from get_actions import enumerate_colorful_actions
import random

# 加载动作全集 M
with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}

def find_entry_by_id(data, target_id):
    """返回匹配 id 的整个 JSON 对象"""
    for entry in data:
        if entry.get("id") == target_id:
            return entry
    return None

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.leaky_relu(out, 0.1)

# 基于残差网络的Actor
class ResNetActor(nn.Module):
    def __init__(self, state_dim=3049, action_dim=action_dim, hidden_dim=1024):
        super().__init__()
        # 输入层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # 残差块（添加Dropout正则化）
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(0.1)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim//2)
        self.drop2 = nn.Dropout(0.1)
        self.res3 = ResidualBlock(hidden_dim//2, hidden_dim//4)
        self.drop3 = nn.Dropout(0.1)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim//4, action_dim)
        nn.init.xavier_uniform_(self.fc_out.weight)
        
    def forward(self, x, mask=None):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        x = self.drop1(self.res1(x))
        x = self.drop2(self.res2(x))
        x = self.drop3(self.res3(x))
        logits = self.fc_out(x)
        if mask is not None:
            # 使用mask过滤非法动作
            logits = logits + (mask - 1) * 1e9
        return F.softmax(logits, dim=-1)

# 基于残差网络的Critic
class ResNetCritic(nn.Module):
    def __init__(self, state_dim=3049, hidden_dim=1024):
        super().__init__()
        # 输入层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # 残差块（增加深度）
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, hidden_dim//2)
        self.res4 = ResidualBlock(hidden_dim//2, hidden_dim//4)
        
        # 价值输出（增加层数）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//8, hidden_dim//16),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//16, 1)
        )
        # 初始化输出层权重
        nn.init.xavier_uniform_(self.value_head[-1].weight, gain=0.01)
        nn.init.constant_(self.value_head[-1].bias, 0.0)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        value = self.value_head(x)
        return value

# 初始化模型（使用GPU优先）
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
print(f"使用设备: {device}")

# 使用残差网络结构
actor = ResNetActor().to(device)
critic = ResNetCritic().to(device)

# 使用带权重衰减的优化器
actor_optimizer = optim.AdamW(actor.parameters(), lr=5e-5, weight_decay=1e-5)
critic_optimizer = optim.AdamW(critic.parameters(), lr=5e-5, weight_decay=1e-5)

# 学习率调度器
actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, 'min', patience=100, factor=0.97, min_lr=1e-5)
critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, 'min', patience=100, factor=0.97, min_lr=1e-5)

gamma = 0.98 # 略微降低折扣因子，平衡即时奖励和长期奖励
gae_lambda = 0.92 # 降低GAE平滑参数，减少优势估计偏差

# 尝试加载已有模型
def load_latest_models(actor, critic, model_dir="models", device=device):
    model_files = sorted(Path(model_dir).glob("actor_ppo_ep*.pth"))
    if model_files:
        latest_actor_path = str(model_files[-1])
        ep = int(latest_actor_path.split("_ep")[1].split(".pth")[0])
        latest_critic_path = f"{model_dir}/critic_ppo_ep{ep}.pth"

        # 加载模型到指定设备
        actor.load_state_dict(torch.load(latest_actor_path, map_location=device))
        print(f"✅ 加载已有 actor 模型: {latest_actor_path}")

        if Path(latest_critic_path).exists():
            critic.load_state_dict(torch.load(latest_critic_path, map_location=device))
            print(f"✅ 加载已有 critic 模型: {latest_critic_path}")
        else:
            print(f"⚠️ 未找到 critic 模型: {latest_critic_path}")

        return ep
    return 0

# 调用加载函数
initial_ep = load_latest_models(actor, critic, device=device)

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算广义优势估计(GAE)
    
    参数:
        rewards (Tensor): 奖励序列
        values (Tensor): 状态价值序列
        dones (Tensor): 终止状态标记(布尔张量)
        gamma (float): 折扣因子
        gae_lambda (float): GAE平滑参数
        
    返回:
        tuple: (advantages, returns) 优势函数和回报
    """
    advantages = []
    gae = 0
    next_value = 0  # 假设最后一步无未来回报

    # 将布尔型dones转换为浮点型(1.0表示终止，0.0表示继续)
    dones_float = dones.float()
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones_float[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones_float[t]) * gae
        advantages.insert(0, gae)
        next_value = values[t]

    advantages = torch.tensor(advantages)
    returns = advantages + values
    # 更稳定的优势标准化
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std < 1e-8:
        advantages = (advantages - adv_mean)
    else:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    return advantages, returns

# 训练函数
def train_on_batch_ppo(batch, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.1, device="cpu", ep=0): #原熵正则系数0.01过低，不利于探索  提高至0.1能更好平衡探索与利用  高熵系数在训练初期尤为重要

    states = torch.tensor(np.array([s["state"] for s in batch]), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array([s["action_id"] for s in batch]), dtype=torch.long).to(device)
    rewards = torch.tensor(np.array([s["reward"] for s in batch]), dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(np.array([s["log_prob"] for s in batch]), dtype=torch.float32).to(device)

    # 自动构造 next_states 和 dones
    next_states = torch.zeros_like(states)
    next_states[:-1] = states[1:]
    next_states[-1] = 0.0
    dones = torch.zeros(len(batch), dtype=torch.bool, device=device)
    dones[-1] = True

    # === Critic 估值 ===
    values = critic(states).squeeze(-1)            # [batch]
    next_values = critic(next_states).squeeze(-1)  # [batch]
    next_values[dones] = 0.0

    # === 计算 GAE 和 Returns ===
    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

    # === PPO 核心：Clipped Surrogate Objective ===
    probs = actor(states)                          # [batch, action_dim]
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()                # 熵正则项

    # 改进的熵系数调整：前期保持高探索，后期缓慢衰减
    if ep < 1000:
        entropy_coef = 0.08  # 前期保持较高探索
    else:
        entropy_coef = max(0.03, 0.08 * (0.995 ** ((ep-1000)//100)))  # 后期缓慢衰减
    
    # 策略比率和裁剪损失
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Critic 损失（值函数 clipping 可选）
    # 使用MSE损失并添加价值函数正则化
    value_clipped = values + (returns - values).clamp(-clip_epsilon, clip_epsilon)
    critic_loss1 = F.smooth_l1_loss(values, returns)
    critic_loss2 = F.smooth_l1_loss(values, value_clipped)
    critic_loss = torch.max(critic_loss1, critic_loss2).mean()
    
    # 添加价值函数正则化项（防止过拟合）
    value_reg = 0.001 * (values ** 2).mean()
    critic_loss += value_reg

    # 总损失
    total_loss = policy_loss + 0.5 * critic_loss - entropy_coef * entropy

    # 更新参数
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪防止爆炸
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    
    actor_optimizer.step()
    critic_optimizer.step()

    return policy_loss.item(), critic_loss.item(), entropy.item()

# 模拟训练流程
def run_training(episodes=3000):
    os.makedirs("models", exist_ok=True)
    for ep in range(initial_ep, initial_ep + episodes): # 从上次的ep继续
        game = GuandanGame(verbose=False)
        memory = []
        game.log(f"\n🎮 游戏开始！当前级牌：{RANKS[game.active_level - 2]}")

        while True:
            if game.is_game_over or len(game.history) > 200:  # 如果游戏结束，立即跳出循环
                break
            player = game.players[game.current_player]
            active_players = 4 - len(game.ranking)

            # **如果 Pass 的人 == "当前有手牌的玩家数 - 1"，就重置轮次**
            if game.pass_count >= (active_players - 1) and game.current_player not in game.ranking:
                if game.jiefeng:
                    first_player = game.ranking[-1]
                    teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1
                    game.log(f"\n🆕 轮次重置！玩家 {teammate + 1} 接风。\n")
                    game.recent_actions[game.current_player] = []  # 记录空列表
                    game.current_player = (game.current_player + 1) % 4
                    game.last_play = None  # ✅ 允许新的自由出牌
                    game.pass_count = 0  # ✅ Pass 计数归零
                    game.is_free_turn = True
                    game.jiefeng = False
                else:
                    game.log(f"\n🆕 轮次重置！玩家 {game.current_player + 1} 可以自由出牌。\n")
                    game.last_play = None  # ✅ 允许新的自由出牌
                    game.pass_count = 0  # ✅ Pass 计数归零
                    game.is_free_turn = True

            if game.current_player == 0:
                # 1. 模型推理（在指定设备上）
                state = game._get_obs()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mask = torch.tensor(game.get_valid_action_mask(player.hand, M, game.active_level, game.last_play), 
                                   dtype=torch.float32).unsqueeze(0).to(device)
                mask = mask.squeeze(0)
                
                # 临时切换到评估模式处理单个样本
                actor.eval()
                with torch.no_grad():
                    probs = actor(state_tensor, mask)
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    action_id = action.item()
                    
                actor.train()  # 切换回训练模式
                action_struct = M_id_dict[action_id]
                    
                # 2. 枚举所有合法出牌组合（带花色）
                combos = enumerate_colorful_actions(action_struct, player.hand, game.active_level)
                if combos:
                    chosen_move = random.choice(combos)
                    if not chosen_move:
                        game.log(f"玩家 {game.current_player + 1} Pass")
                        game.pass_count += 1
                        game.recent_actions[game.current_player] = ['Pass']  # 记录 Pass
                    else:
                        # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                        game.last_play = chosen_move
                        game.last_player = game.current_player
                        for card in chosen_move:
                            player.played_cards.append(card)
                            player.hand.remove(card)
                        game.log(f"玩家 {game.current_player + 1} 出牌: {' '.join(chosen_move)}")
                        game.recent_actions[game.current_player] = list(chosen_move)  # 记录出牌
                        game.jiefeng = False
                        if not player.hand:  # 玩家出完牌
                            game.log(f"\n🎉 玩家 {game.current_player + 1} 出完所有牌！\n")
                            game.ranking.append(game.current_player)
                            if len(game.ranking) <= 2:
                                game.jiefeng = True

                        game.pass_count = 0
                        if not player.hand:
                            game.pass_count -= 1

                        if game.is_free_turn:
                            game.is_free_turn = False
                else:
                    game.log(f"玩家 {game.current_player + 1} Pass")
                    game.pass_count += 1
                    game.recent_actions[game.current_player] = ['Pass']  # 记录 Pass

                # 强化奖励信号设计
                entry = find_entry_by_id(M, action_id)
                reward = 0
                
                # 改进的奖励函数设计
                reward = 0
                hand_size_before = len(player.hand)
                
                # 合法动作基础奖励
                if mask[action_id] and action_id != 0:  # 非Pass动作
                    # 基础奖励 = 牌型点数 * 逻辑点倒数
                    base_reward = float(len(entry['points'])) * (1 / entry['logic_point'])
                    
                    # 炸弹类牌型额外奖励（使用平方根缩放）
                    if 120 <= action_id <= 364: 
                        bomb_strength = min(math.sqrt(entry['logic_point']), 2.0)  # 平方根缩放
                        base_reward *= 1.0 + bomb_strength
                    
                    # 手牌减少奖励（线性+阶段奖励）
                    cards_played = hand_size_before - len(player.hand)
                    hand_reward = 0.03 * cards_played
                    
                    # 阶段奖励：根据剩余手牌数给予额外奖励
                    if len(player.hand) <= 5:
                        hand_reward += 0.1 * (10 - len(player.hand))
                    
                    reward = base_reward + hand_reward
                    
                # 非法动作惩罚（基于阶段）
                elif not mask[action_id]:
                    # 惩罚随游戏阶段增加
                    penalty_factor = 1.0 + (20 - hand_size_before) * 0.05
                    reward = -1.0 * penalty_factor
                
                memory.append({"state": state, "action_id": action_id, "reward": reward,"log_prob": log_prob.item()})
                player.last_played_cards = game.recent_actions[game.current_player]
                game.current_player = (game.current_player + 1) % 4
            else:
                game.ai_play(player)  # 其他人用随机
            # **记录最近 5 轮历史**记录历史（每轮结束时）
            if game.current_player == 0 and any(action != ['None'] for action in game.recent_actions):
                round_history = [game.recent_actions[i].copy() for i in range(4)]
                game.history.append(round_history)
                
                # 重置最近动作（使用深拷贝避免引用问题）
                game.recent_actions = [['None'] for _ in range(4)]

        # 团队奖励：使用分段函数
        if game.upgrade_amount > 0:
            team_reward = min(5.0, game.upgrade_amount * 1.5)  # 上限5.0
        else:
            team_reward = 0.0
        
        # 根据玩家位置分配奖励（玩家0属于0队，队友是玩家2）
        player_position = 0
        team_id = 0 if player_position in [0, 2] else 1
        team_multiplier = 1 if team_id == 0 else -1
        
        if memory:  # 确保 memory 不为空
            # 将团队奖励分配给整个episode
            for i in range(len(memory)):
                # 改进的奖励分配：基于时间步的线性衰减
                decay = max(0.2, 1.0 - i / len(memory))  # 线性衰减，最小保留20%
                memory[i]["reward"] += team_multiplier * team_reward * decay
                
            al, cl, entropy = train_on_batch_ppo(memory, entropy_coef=0.1, device=device, ep=ep)
            
            # 使用损失更新学习率调度器
            actor_scheduler.step(al)
            critic_scheduler.step(cl)
            
            if (ep + 1) % 50 == 0:
                # 计算奖励统计
                rewards = [s["reward"] for s in memory]
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                max_reward = max(rewards) if rewards else 0
                min_reward = min(rewards) if rewards else 0
                
                # 计算策略熵（使用memory中的状态）
                if memory:
                    # 从memory中提取所有状态
                    states = torch.tensor(np.array([s["state"] for s in memory]), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        probs = actor(states)
                        dist = Categorical(probs)
                        policy_entropy = dist.entropy().mean().item()
                else:
                    policy_entropy = 0.0
                
                # 计算动作分布熵
                action_probs = probs.mean(dim=0)
                action_entropy = Categorical(action_probs).entropy().item()
                
                print(f"Episode {ep + 1}: "
                      f"ALoss={al:.4f}, CLoss={cl:.4f}, "
                      f"Entropy={entropy:.4f}, ActionEntropy={action_entropy:.4f}, "
                      f"Reward(avg={avg_reward:.2f}, max={max_reward:.2f}, min={min_reward:.2f}), "
                      f"LR(actor={actor_optimizer.param_groups[0]['lr']:.2e}, critic={critic_optimizer.param_groups[0]['lr']:.2e})")

        if (ep + 1) % 200 == 0:
            torch.save(actor.state_dict(), f"models/actor_ppo_ep{ep + 1}.pth")
            torch.save(critic.state_dict(), f"models/critic_ppo_ep{ep + 1}.pth")

if __name__ == "__main__":
    run_training()

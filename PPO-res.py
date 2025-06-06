"""
PPO.py
掼蛋Proximal Policy Optimization (PPO) 强化学习智能体训练代码。
作者: xxyyboy
日期: 2025-06-06
"""
import os,sys
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# 加载动作配置
try:
    with open("guandan_actions.json", "r", encoding="utf-8") as f:
        M = json.load(f)
    action_dim = len(M)
    M_id_dict = {a['id']: a for a in M}
except FileNotFoundError:
    logging.error("未找到guandan_actions.json文件")
    raise

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.leaky_relu(out, 0.1)

# 改进的Actor网络
class ResNetActor(nn.Module):
    def __init__(self, state_dim=3049, action_dim=action_dim, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim//2),
            ResidualBlock(hidden_dim//2, hidden_dim//4)
        ])
        self.fc_out = nn.Linear(hidden_dim//4, action_dim)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x, mask=None):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        for res_block in self.res_blocks:
            x = res_block(x)
        logits = self.fc_out(x)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        return F.softmax(logits, dim=-1)

# 改进的Critic网络
class ResNetCritic(nn.Module):
    def __init__(self, state_dim=3049, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim//2),
            ResidualBlock(hidden_dim//2, hidden_dim//4),
            ResidualBlock(hidden_dim//4, hidden_dim//8)
        ])
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim//8, hidden_dim//16),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim//16, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.value_head(x)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

# 初始化模型
actor = ResNetActor().to(device)
critic = ResNetCritic().to(device)

# 优化器配置
actor_optimizer = optim.AdamW(actor.parameters(), lr=5e-5, weight_decay=1e-5)
critic_optimizer = optim.AdamW(critic.parameters(), lr=5e-5, weight_decay=1e-5)

# 学习率调度器
actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    actor_optimizer, 'min', patience=100, factor=0.97, min_lr=1e-5
)
critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    critic_optimizer, 'min', patience=100, factor=0.97, min_lr=1e-5
)

def calculate_reward(entry, player, mask, action_id, hand_size_before):
    """改进的奖励计算函数"""
    reward = 0
    
    if mask[action_id] and action_id != 0:
        # 基础奖励
        base_reward = float(len(entry['points'])) * (1 / max(1, entry['logic_point']))
        
        # 炸弹奖励
        bomb_types = {
            'bomb': (120, 364),
            'joint_bomb': (365, 455),
            'suite_bomb': (456, 495)
        }
        
        for bomb_type, (start, end) in bomb_types.items():
            if start <= action_id <= end:
                multiplier = 2.0 if bomb_type == 'suite_bomb' else 1.5 if bomb_type == 'joint_bomb' else 1.2
                base_reward *= multiplier
                break
                
        # 手牌减少奖励
        cards_played = hand_size_before - len(player.hand)
        hand_reward = 0.05 * math.log(cards_played + 1)
        
        # 关键阶段奖励
        remaining_cards = len(player.hand)
        if remaining_cards <= 5:
            stage_reward = 0.2 * (6 - remaining_cards)
            hand_reward += stage_reward
            
        reward = base_reward + hand_reward
        
    elif not mask[action_id]:
        game_progress = 1.0 - (hand_size_before / 27)
        penalty = -1.0 * (1.0 + game_progress)
        reward = penalty
        
    return reward

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """优化的GAE计算"""
    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0
    next_value = 0
    
    dones_float = dones.float()
    
    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones_float[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones_float[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns

def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, ep, model_dir="models"):
    """保存训练检查点"""
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'episode': ep
    }
    torch.save(checkpoint, f"{model_dir}/checkpoint_ep{ep}.pth")
    logging.info(f"保存检查点: checkpoint_ep{ep}.pth")

def load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, model_dir="models", device=device):
    """加载训练检查点"""
    model_files = sorted(Path(model_dir).glob("checkpoint_ep*.pth"))
    if not model_files:
        logging.info("未找到检查点文件，从头开始训练")
        return 0
        
    latest_checkpoint = str(model_files[-1])
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    ep = checkpoint['episode']
    logging.info(f"加载检查点: {latest_checkpoint}")
    return ep

def train_on_batch_ppo(batch, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.1, device=device, ep=0):
    """PPO批次训练"""
    states = torch.tensor(np.array([s["state"] for s in batch]),dtype=torch.float32).to(device)
    actions = torch.tensor(np.array([s["action_id"] for s in batch]),dtype=torch.long).to(device)
    rewards = torch.tensor(np.array([s["reward"] for s in batch]),dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(np.array([s["log_prob"] for s in batch]),dtype=torch.float32).to(device)
    dones = torch.tensor(np.array([s["done"] for s in batch]),dtype=torch.bool).to(device)

    # Critic评估
    values = critic(states).squeeze(-1)
    next_states = torch.zeros_like(states)
    next_states[:-1] = states[1:]
    next_values = critic(next_states).squeeze(-1)
    next_values[dones] = 0.0

    # 计算GAE和回报
    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

    # PPO更新
    probs = actor(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # 自适应熵系数
    if ep < 1000:
        entropy_coef = 0.08
    else:
        entropy_coef = max(0.03, 0.08 * (0.995 ** ((ep-1000)//100)))

    # 策略损失
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Critic损失
    value_clipped = values + (returns - values).clamp(-clip_epsilon, clip_epsilon)
    critic_loss1 = F.smooth_l1_loss(values, returns)
    critic_loss2 = F.smooth_l1_loss(value_clipped, returns)
    critic_loss = torch.max(critic_loss1, critic_loss2).mean()

    # 价值函数正则化
    value_reg = 0.001 * (values ** 2).mean()
    critic_loss += value_reg

    # 总损失
    total_loss = policy_loss + 0.5 * critic_loss - entropy_coef * entropy

    # 更新参数
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    total_loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

    actor_optimizer.step()
    critic_optimizer.step()

    return policy_loss.item(), critic_loss.item(), entropy.item()

def run_training(episodes=3000):
    """主训练循环"""
    os.makedirs("models", exist_ok=True)
    writer = SummaryWriter(f'runs/guandan_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # 加载检查点
    initial_ep = load_checkpoint(actor, critic, actor_optimizer, critic_optimizer)
    best_reward = float('-inf')
    
    try:
        for ep in range(initial_ep, initial_ep + episodes):
            game = GuandanGame(verbose=True)
            memory = []
            episode_reward = 0
            
            while not game.is_game_over and len(game.history) <= 200:
                if game.current_player == 0:
                    state = game._get_obs()
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    mask = torch.tensor(game.get_valid_action_mask(game.players[0].hand, M,game.active_level, game.last_play),dtype=torch.float32).unsqueeze(0).to(device)
                    mask = mask.squeeze(0)
                    
                    # 模型推理
                    actor.eval()
                    with torch.no_grad():
                        probs = actor(state_tensor, mask)
                        dist = Categorical(probs)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        action_id = action.item()
                    actor.train()
                    
                    # 执行动作
                    entry = M_id_dict[action_id]
                    player = game.players[0]
                    hand_size_before = len(player.hand)
                    
                    # 枚举合法动作
                    combos = enumerate_colorful_actions(entry, player.hand, game.active_level)
                    
                    if combos:
                        chosen_move = random.choice(combos)
                        if not chosen_move:
                            game.log(f"玩家 1 Pass")
                            game.pass_count += 1
                            game.recent_actions[0] = ['Pass']
                        else:
                            game.last_play = chosen_move
                            game.last_player = 0
                            for card in chosen_move:
                                player.played_cards.append(card)
                                player.hand.remove(card)
                            game.log(f"玩家 1 出牌: {' '.join(chosen_move)}")
                            game.recent_actions[0] = list(chosen_move)
                            game.jiefeng = False
                            
                            if not player.hand:
                                game.log(f"\n🎉 玩家 1 出完所有牌！\n")
                                game.ranking.append(0)
                                if len(game.ranking) <= 2:
                                    game.jiefeng = True
                                    
                            game.pass_count = 0
                            if not player.hand:
                                game.pass_count -= 1
                                
                            if game.is_free_turn:
                                game.is_free_turn = False
                    else:
                        game.log(f"玩家 1 Pass")
                        game.pass_count += 1
                        game.recent_actions[0] = ['Pass']
                    
                    # 计算奖励
                    reward = calculate_reward(entry, player, mask, action_id, hand_size_before)
                    episode_reward += reward
                    
                    # 记录转换
                    memory.append({
                        "state": state,
                        "action_id": action_id,
                        "reward": reward,
                        "log_prob": log_prob.item(),
                        "done": game.is_game_over
                    })
                    
                    player.last_played_cards = game.recent_actions[0]
                    game.current_player = (game.current_player + 1) % 4
                else:
                    game.ai_play(game.players[game.current_player])
                
                # 更新历史记录
                if game.current_player == 0 and any(action != ['None'] for action in game.recent_actions):
                    round_history = [action.copy() for action in game.recent_actions]
                    game.history.append(round_history)
                    game.recent_actions = [['None'] for _ in range(4)]
            
            # 训练更新
            if memory:
                policy_loss, value_loss, entropy = train_on_batch_ppo(
                    memory, entropy_coef=0.1, device=device, ep=ep
                )
                
                # 更新学习率
                actor_scheduler.step(policy_loss)
                critic_scheduler.step(value_loss)
                
                # 记录指标
                writer.add_scalar('Training/PolicyLoss', policy_loss, ep)
                writer.add_scalar('Training/ValueLoss', value_loss, ep)
                writer.add_scalar('Training/Entropy', entropy, ep)
                writer.add_scalar('Training/EpisodeReward', episode_reward, ep)
                writer.add_scalar('Training/ActorLR', actor_optimizer.param_groups[0]['lr'], ep)
                writer.add_scalar('Training/CriticLR', critic_optimizer.param_groups[0]['lr'], ep)
                
                # 打印训练信息
                if (ep + 1) % 50 == 0:
                    logging.info(
                        f"Episode {ep + 1}: "
                        f"PLoss={policy_loss:.4f}, VLoss={value_loss:.4f}, "
                        f"Entropy={entropy:.4f}, Reward={episode_reward:.2f}, "
                        f"LR_actor={actor_optimizer.param_groups[0]['lr']:.2e}, "
                        f"LR_critic={critic_optimizer.param_groups[0]['lr']:.2e}"
                    )
                
                # 保存最佳模型
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, ep + 1,
                                 model_dir="models/best")
            
            # 定期保存检查点
            if (ep + 1) % 200 == 0:
                save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, ep + 1)
                
    except KeyboardInterrupt:
        logging.info("训练被手动中断")
        save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, ep + 1,
                     model_dir="models/interrupted")
    finally:
        writer.close()

if __name__ == "__main__":
    logging.info("开始训练 掼蛋 PPO 智能体")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"设备: {device}")
    run_training()

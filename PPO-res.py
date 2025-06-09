"""
改进的PPO算法实现 - 掼蛋纸牌游戏AI训练 20250609
核心目标：降低AI选择Pass的概率，鼓励积极出牌
"""

import os
import sys
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
from copy import deepcopy

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

class ReplayBuffer:
    """存储训练样本的循环缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = []  # 存储样本的列表
        self.capacity = capacity # 最大容量
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, log_prob):
        """添加新样本"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """随机采样一批样本"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, log_probs = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones, log_probs
        
    def __len__(self):
        return len(self.buffer)

class ResidualBlock(nn.Module):
    """残差块结构，缓解深层网络梯度消失"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential() # 捷径连接
        
        if in_dim != out_dim: # 维度不匹配时使用投影
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LayerNorm(out_dim)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.leaky_relu(self.ln1(self.bn1(self.fc1(x))), 0.1)
        out = self.dropout(out)
        out = self.ln2(self.bn2(self.fc2(out)))
        out += residual
        return F.leaky_relu(out, 0.1)

class SharedBackbone(nn.Module):
    """共享特征提取网络"""
    def __init__(self, state_dim=3049, hidden_dim=2048):  # 增大隐藏层维度
        super().__init__()
        # 手牌特征编码器 (108维)
        self.card_encoder = nn.Sequential(
            nn.Linear(108, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512)
        )
        
        # 历史动作时序编码器 (2160维)
        self.history_encoder = nn.LSTM(
            input_size=2160, 
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        # 级牌特征嵌入 (13维)
        self.level_embed = nn.Embedding(13, 64)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(512+512+64, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim//2)
        ])
        self.eval_mode = False  # 初始化eval_mode属性
        self._init_weights() # 正交初始化
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x):
        # 手牌特征提取
        card_feat = self.card_encoder(x[..., :108])
        
        # 历史动作特征提取
        history_feat, _ = self.history_encoder(x[..., 108:2268].unsqueeze(1))
        history_feat = history_feat.squeeze(1)
        
        # 级牌特征提取
        level_idx = torch.argmax(x[..., 2268:2281], dim=-1).long()
        level_feat = self.level_embed(level_idx)
        
        # 特征融合
        x = torch.cat([card_feat, history_feat, level_feat], dim=-1)
        x = self.fusion(x)
        
        # 残差块处理
        for res_block in self.res_blocks:
            x = res_block(x)
        return x
    
    def eval(self):
        """设置评估模式"""
        super().eval()
        self.eval_mode = True
        
    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)
        self.eval_mode = not mode
        
class ImprovedResNetActor(nn.Module):
    """策略网络"""
    def __init__(self, backbone, action_dim=action_dim):
        super().__init__()
        self.backbone = backbone # 共享特征提取
        backbone_out_dim = backbone.res_blocks[-1].fc2.out_features
        # 策略头
        self.fc_policy = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim//2),
            nn.LayerNorm(backbone_out_dim//2),
            nn.ReLU(),
            nn.Linear(backbone_out_dim//2, action_dim),
            nn.LayerNorm(action_dim)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x, mask=None):
        features = self.backbone(x)
        logits = self.fc_policy(features)
        # 重点：降低Pass（假设为动作0）的logit分数，训练初期出牌更积极
        logits[..., 0] -= 1.5
        if mask is not None:
            logits = logits + (mask.float() - 1) * 1e9
        probs = F.softmax(logits, dim=-1)
        return probs, logits

class ImprovedResNetCritic(nn.Module):
    """价值网络"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        backbone_out_dim = backbone.res_blocks[-1].fc2.out_features
        self.fc_value = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim//2),
            nn.LayerNorm(backbone_out_dim//2),
            nn.ReLU(),
            nn.Linear(backbone_out_dim//2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() # 输出在[-1,1]范围
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        features = self.backbone(x)
        value = self.fc_value(features)
        return value

def select_action(actor, state, mask, device, is_free_turn, ep):
    """智能动作选择策略3.0"""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 动态探索参数
    explore_factor = max(0.1, 0.5 * (0.98 ** (ep // 100)))  # 衰减更慢
    temp = max(0.5, 1.5 - ep/4000)  # 温度参数
    
    with torch.no_grad():
        probs, logits = actor(state_tensor, mask_tensor)
        
        # 强化Pass抑制
        if is_free_turn:
            logits[0, 0] = -float('inf')  # 自由出牌禁止Pass
        else:
            # 动态Pass惩罚 = 基础值 + 训练进度调整
            pass_penalty = 1.5 + (1 - ep/6000)
            logits[0, 0] -= pass_penalty
            
        # 炸弹动作奖励
        bomb_mask = torch.zeros_like(logits)
        bomb_mask[:, 120:456] = 1  # 炸弹动作区间
        logits += bomb_mask * (0.5 + ep/8000)  # 随训练逐渐重视炸弹
        
        # 探索机制
        if random.random() < explore_factor:
            # 优先探索非Pass动作
            valid_actions = torch.nonzero(mask_tensor[0] * (torch.arange(mask_tensor.size(1)) != 0))
            if len(valid_actions) > 0:
                action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                return action.item()

def validate_game_state(game):
    """验证游戏状态的合法性"""
    # 检查出牌权
    if game.is_free_turn and game.last_play:
        logging.warning("状态错误：自由出牌回合存在last_play")
        game.last_play = []
        
    # 检查排名
    if len(set(game.ranking)) != len(game.ranking):
        logging.error("状态错误：排名重复")
        game.ranking = list(dict.fromkeys(game.ranking))
        
    # 检查Pass计数
    if game.pass_count > 4:
        logging.warning("状态错误：pass_count超过4")
        game.pass_count = 0
        
    return True

def calculate_improved_reward(entry, player, mask, action_id, hand_size_before, game, ep):
    """动态奖励函数2.0 - 强化关键决策奖励"""
    reward = 0.0
    hand_size = len(player.hand)
    progress = (27 - hand_size) / 27  # 游戏进度[0,1]
    
    # 1. 智能Pass惩罚机制
    if action_id == 0:  # Pass动作
        # 动态惩罚系数 = 基础惩罚 + 进度惩罚 + 训练阶段衰减
        base_penalty = 0.8 + (1 - progress)**2  # 后期惩罚加重
        phase_factor = max(0.3, 1.0 - ep/5000)  # 训练后期降低惩罚
        penalty = -base_penalty * phase_factor
        
        # 自由出牌回合禁止Pass
        if game.is_free_turn:
            penalty *= 2.0
            
        # 对手即将出完牌时加重惩罚
        opp_hands = [len(p.hand) for i,p in enumerate(game.players) if i != 0]
        if min(opp_hands) <= 3:
            penalty *= 1.5 + (4 - min(opp_hands))*0.3
            
        reward += penalty
        
    # 2. 出牌奖励体系
    else:
        # 基础奖励 (动态调整)
        base_reward = 0.5 + progress*0.5  # 后期奖励更高
        
        # 牌型系数
        action_type = entry.get('type', '')
        type_bonus = {
            'single': 0.1, 'pair': 0.2, 'trio': 0.3,
            'bomb': 1.2, 'rocket': 2.0, 'sequence': 0.4
        }.get(action_type, 0.0)
        
        # 数量奖励 (非线性增长)
        cards_played = hand_size_before - hand_size
        count_bonus = 0.2 * math.sqrt(cards_played)
        
        reward += base_reward + type_bonus + count_bonus
        
        # 关键阶段奖励 (剩余牌<5时指数增长)
        if hand_size <= 5:
            reward += (6 - hand_size)**1.5 * 0.2
            
    # 3. 终局奖励优化
    if game.is_game_over:
        rank = game.ranking.index(0)  # 玩家名次
        rank_rewards = [10.0, 7.0, -4.0, -6.0]  # 更平缓的奖励曲线
        
        # 队伍胜利检测
        teammate = 2 if 0 in [0,2] else 3
        if rank < 2 and teammate in game.ranking[:2]:
            rank_rewards[rank] *= 1.5  # 队伍胜利加成
            
        reward += rank_rewards[rank]
    
    return reward  # 确保函数有返回值

def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t].float()) * gae
        advantages[t] = gae
    returns = advantages + values
    # 添加优势值裁剪和更稳定的标准化
    advantages = torch.clamp(advantages, -3.0, 3.0)
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) + 1e-7
    advantages = (advantages - adv_mean) / adv_std
    return advantages, returns

def save_checkpoint(backbone, actor, critic, optimizer, ep, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'backbone_state_dict': backbone.state_dict(),
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': ep
    }
    torch.save(checkpoint, f"{model_dir}/checkpoint_ep{ep}.pth")
    logging.info(f"保存检查点: checkpoint_ep{ep}.pth")

def load_checkpoint(device, backbone, actor, critic, optimizer, model_dir="models"):
    model_files = sorted(Path(model_dir).glob("checkpoint_ep*.pth"))
    if not model_files:
        logging.info("未找到检查点文件，从头开始训练")
        return 0
    latest_checkpoint = str(model_files[-1])
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ep = checkpoint['episode']
    logging.info(f"加载检查点: {latest_checkpoint}")
    return ep


def train_on_batch_ppo(states, actions, rewards, next_states, dones, old_log_probs,
                      backbone, actor, critic, target_critic, optimizer,
                      gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                      entropy_coef=0.1, device=device, ep=0):
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.BoolTensor(dones).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)
    with torch.no_grad():
        next_values = target_critic(next_states).squeeze(-1)
    values = critic(states).squeeze(-1)
    advantages, returns = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda)
    probs, logits = actor(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    # 调整熵系数
    if ep < 200:
        entropy_coef = 0.05  # 降低初始熵系数
    else:
        entropy_coef = max(0.005, 0.05 * (0.995 ** ((ep-200)//50)))
        
    kl_div = (old_log_probs - new_log_probs).mean()
    
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    # 策略正则化和Pass惩罚
    pass_probs = probs[:, 0]
    policy_loss += 0.1 * pass_probs.mean()
    if kl_div > 0.02:
        policy_loss *= 1.2
    value_clipped = values + (returns - values).clamp(-clip_epsilon, clip_epsilon)
    critic_loss1 = F.smooth_l1_loss(values, returns)
    critic_loss2 = F.smooth_l1_loss(value_clipped, returns)
    critic_loss = torch.max(critic_loss1, critic_loss2).mean()
    value_loss_weight = max(0.25, 0.5 * (0.999 ** ep))
    total_loss = policy_loss + value_loss_weight * critic_loss - entropy_coef * entropy
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    optimizer.step()
    return policy_loss.item(), critic_loss.item(), entropy.item(), kl_div.item()

# ...其余内容不变...

def run_training(episodes=30000):
    """改进的训练流程 - 添加课程学习和目标网络"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 课程学习配置
    curriculum = {
        0: {'level': (2,5), 'opponent': 'random'},
        5000: {'level': (5,8), 'opponent': 'rule_based'},
        10000: {'level': (8,14), 'opponent': 'self'}
    }
    
    # 初始化网络
    backbone = SharedBackbone().to(device)
    actor = ImprovedResNetActor(backbone).to(device)
    critic = ImprovedResNetCritic(backbone).to(device)
    
    # 初始化目标网络（用于稳定训练）
    target_backbone = SharedBackbone().to(device)
    target_critic = ImprovedResNetCritic(target_backbone).to(device)
    
    # 同步初始参数
    target_backbone.load_state_dict(backbone.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # 优化器分组设置
    optimizer_params = []
    
    # 1. 添加actor独有参数
    optimizer_params.append({
        'params': [p for n,p in actor.named_parameters() 
                  if not n.startswith('backbone.')],
        'lr': 3e-5
    })
    
    # 2. 添加critic独有参数
    optimizer_params.append({
        'params': [p for n,p in critic.named_parameters()
                  if not n.startswith('backbone.')],
        'lr': 3e-5
    })
    
    # 3. 添加共享主干参数（只添加一次）
    optimizer_params.append({
        'params': backbone.parameters(),
        'lr': 1e-5
    })
    
    optimizer = optim.AdamW(optimizer_params, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=50, factor=0.97, min_lr=1e-5
    )
    memory = ReplayBuffer(capacity=10000)
    writer = SummaryWriter(f'runs/guandan_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    initial_ep = load_checkpoint(device, backbone, actor, critic, optimizer)
    best_reward = float('-inf')
    def soft_update(target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    try:
        for ep in range(initial_ep, initial_ep + episodes):
            game = GuandanGame(verbose=True)
            episode_reward = 0
            episode_steps = 0
            pass_penalty_given = False  # 防止多次惩罚
            while not game.is_game_over and len(game.history) <= 200:
                # 强化连续Pass检测和处理
                if game.pass_count >= 2:  # 降低检测阈值
                    # 记录连续Pass轮次
                    consecutive_pass_rounds = game.pass_count
                    
                    # 重置出牌权给最后出牌的玩家
                    if game.last_player is not None:
                        game.current_player = game.last_player
                        game.log(f"玩家 {game.last_player + 1} 获得新一轮出牌权（连续{consecutive_pass_rounds}次Pass后）")
                    
                    # 应用全局惩罚（每轮Pass都惩罚）
                    global_penalty = -0.5 * consecutive_pass_rounds
                    if len(memory) > 0:
                        memory.push(memory.buffer[-1][0], 0, global_penalty, 
                                  memory.buffer[-1][0], False, 0.0)
                        episode_reward += global_penalty
                    
                    # 重置状态
                    game.last_play = []
                    game.pass_count = 0
                    game.recent_actions = [['None'] for _ in range(4)]
                    continue
    
                # 检查连续4次Pass惩罚
                if game.pass_count >= 4 and not pass_penalty_given:
                    # 对所有玩家（这里只训练主智能体，reward记录在memory）
                    penalty = -2.0
                    # 仅对主智能体做记录
                    if len(memory) > 0:
                        memory.push(memory.buffer[-1][0], 0, penalty, memory.buffer[-1][0], False, 0.0)
                        episode_reward += penalty
                    # 日志提示
                    logging.info("所有玩家连续4次Pass，给予所有玩家惩罚！")
                    # 清空pass_count和recent_actions，进入新一轮
                    game.pass_count = 0
                    game.recent_actions = [['None'] for _ in range(4)]
                    pass_penalty_given = True
                else:
                    pass_penalty_given = False

                if game.current_player == 0:
                    state = game._get_obs()
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    mask = torch.tensor(game.get_valid_action_mask(game.players[0].hand, M,
                                                 game.active_level, game.last_play),
                        dtype=torch.float32).unsqueeze(0).to(device)
                    mask = mask.squeeze(0)
                    actor.eval()
                    with torch.no_grad():
                        probs, _ = actor(state_tensor, mask)
                        temperature = 1.1
                        adj_probs = (probs ** (1/temperature))
                        adj_probs = adj_probs / adj_probs.sum()
                        dist = Categorical(adj_probs)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        action_id = action.item()
                    actor.train()
                    entry = M_id_dict[action_id]
                    player = game.players[0]
                    hand_size_before = len(player.hand)
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
                                game.ranking.append(0)  # 玩家0的索引直接使用0
                                # 立即结束当前玩家回合
                                game.is_game_over = len(game.ranking) >= 4
                                game.pass_count = 0
                                break  # 跳出当前回合循环
                                
                            game.pass_count = 0
                            # 移除重复的hand检查
                            if game.is_free_turn:
                                game.is_free_turn = False
                    else:
                        game.log(f"玩家 1 Pass")
                        game.pass_count += 1
                        game.recent_actions[0] = ['Pass']
                    next_state = game._get_obs()
                    reward = calculate_improved_reward(entry, player, mask, action_id, 
                                                    hand_size_before, game, ep)
                    episode_reward += reward
                    memory.push(state, action_id, reward, next_state, game.is_game_over, log_prob.item())
                    player.last_played_cards = game.recent_actions[0]
                    game.current_player = (game.current_player + 1) % 4 
                    episode_steps += 1
                else:
                    # 添加AI玩家结束检查
                    current_ai_player = game.players[game.current_player]
                    game.ai_play(current_ai_player)
                    
                    # 检查AI玩家是否出完牌（使用玩家索引代替seat属性）
                    if not current_ai_player.hand and game.current_player not in game.ranking:
                        game.ranking.append(game.current_player)
                        game.is_game_over = len(game.ranking) >= 4
                        if game.is_game_over:
                            break  # 立即结束游戏循环
                round_history = []
                if game.current_player == 0 and any(action != ['None'] for action in game.recent_actions):
                    round_history = [action.copy() for action in game.recent_actions]
                    game.history.append(round_history) 
                    game.recent_actions = [['None'] for _ in range(4)]
                
                all_others_pass = False    
                if len(round_history) == 4 and round_history[0] != ['Pass']:
                    all_others_pass = all(action == ['Pass'] for action in round_history[1:4])
                    
                if all_others_pass:
                    game.is_free_turn = True
                    game.last_play = []
                    game.pass_count = 0
                    if hasattr(game, 'last_player'):
                        game.current_player = game.last_player
                    print("自由出牌轮，mask:", mask.cpu().numpy())

                if all(action == ['Pass'] for action in game.recent_actions):
                    game.is_free_turn = True
                    game.last_play = []
                    game.pass_count = 0
                    print("死循环检测，mask:", mask.cpu().numpy())
        
            if len(memory) >= 128:
                states, actions, rewards, next_states, dones, old_log_probs = memory.sample(128)
                policy_loss, value_loss, entropy, kl_div = train_on_batch_ppo(
                    states, actions, rewards, next_states, dones, old_log_probs,
                    backbone, actor, critic, target_critic, optimizer,
                    entropy_coef=0.1, device=device, ep=ep
                )
                soft_update(target_backbone, backbone)
                soft_update(target_critic, critic)
                scheduler.step(policy_loss)
                writer.add_scalar('Training/PolicyLoss', policy_loss, ep)
                writer.add_scalar('Training/ValueLoss', value_loss, ep)
                writer.add_scalar('Training/Entropy', entropy, ep)
                writer.add_scalar('Training/KLDivergence', kl_div, ep)
                writer.add_scalar('Training/EpisodeReward', episode_reward, ep)
                writer.add_scalar('Training/EpisodeSteps', episode_steps, ep)
                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Training/LR_group_{i}', param_group['lr'], ep)
                if (ep + 1) % 50 == 0:
                    logging.info(
                        f"Episode {ep + 1}: "
                        f"PLoss={policy_loss:.4f}, VLoss={value_loss:.4f}, "
                        f"Entropy={entropy:.4f}, KL={kl_div:.4f}, "
                        f"Reward={episode_reward:.2f}, Steps={episode_steps}"
                    )
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    save_checkpoint(backbone, actor, critic, optimizer, ep + 1,
                                 model_dir="models/best")
            if (ep + 1) % 200 == 0:
                save_checkpoint(backbone, actor, critic, optimizer, ep + 1)
    except KeyboardInterrupt:
        logging.info("训练被手动中断")
        save_checkpoint(backbone, actor, critic, optimizer, ep + 1,
                      model_dir="models/interrupted")
    finally:
        writer.close()

if __name__ == "__main__":
    logging.info("开始训练 掼蛋 PPO 智能体")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"设备: {device}")
    run_training()

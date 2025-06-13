"""
改进的PPO算法实现 - 掼蛋纸牌游戏AI训练
核心目标：降低AI选择Pass的概率，鼓励积极出牌
"""

import os
import sys
import time
import math
import threading
import psutil
import os
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
    def __init__(self, capacity=20000):
        self.buffer = []  # 存储样本的列表
        self.capacity = capacity # 最大容量
        self.position = 0
        self.lock = threading.Lock()  # 添加线程锁
        
    def push(self, state, action, reward, next_state, done, log_prob):
        """添加新样本"""
        # 确保数据有效
        if state is None or next_state is None or log_prob is None:
            return
        
        # 使用线程锁确保线程安全
        with self.lock:
            # 当缓冲区未满时，直接添加新元素
            if len(self.buffer) < self.capacity:
                self.buffer.append((state, action, reward, next_state, done, log_prob))
            else:
                # 缓冲区已满时，覆盖最旧的数据
                self.buffer[self.position] = (state, action, reward, next_state, done, log_prob)
            self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """随机采样一批样本"""
        # 过滤掉None值
        valid_buffer = [item for item in self.buffer if item is not None]
        if not valid_buffer:
            # 返回空数组而不是None，避免解包错误
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # 从有效缓冲区采样（避免采样到None）
        batch = random.sample(valid_buffer, min(batch_size, len(valid_buffer)))
        states, actions, rewards, next_states, dones, log_probs = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones, log_probs
        
    def __len__(self):
        return len(self.buffer)

# 修改SharedBackbone类
class SharedBackbone(nn.Module):
    def __init__(self, state_dim=3049, hidden_dim=1024):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(state_dim)
        self.hidden_dim=hidden_dim
        self.eval_mode = False
        
        # 手牌编码器
        self.card_encoder = nn.Sequential(
            nn.Linear(108, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 历史动作编码器
        self.history_encoder = nn.Sequential(
            nn.Linear(2160, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 场景信息编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(781, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,  # 减少注意力头数
            dim_feedforward=1024,  # 减小前馈层维度
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6  # 减少Transformer层数
        )
        
        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, 3, hidden_dim) * 0.02
        )
        
        # 分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # 输出投影层
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.input_bn(x)
        
        # 提取各特征
        card_feat = self.card_encoder(x[..., :108])  # [B, 512]
        history_feat = self.history_encoder(x[..., 108:2268])  # [B, 512]
        context_feat = self.context_encoder(x[..., 2268:])  # [B, 256]
        
        # 投影到统一维度
        card_feat = F.linear(card_feat, torch.zeros(self.hidden_dim, 512).to(x.device))
        history_feat = F.linear(history_feat, torch.zeros(self.hidden_dim, 512).to(x.device))
        context_feat = F.linear(context_feat, torch.zeros(self.hidden_dim, 256).to(x.device))
        
        # 添加分类token和位置编码
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, D]
        features = torch.stack([card_feat, history_feat, context_feat], dim=1)  # [B, 3, D]
        features = features + self.position_embedding
        features = torch.cat([cls_tokens, features], dim=1)  # [B, 4, D]
        
        # Transformer处理
        features = self.transformer_encoder(features)
        
        # 取分类token作为输出
        out = self.proj(features[:, 0])
        
        return out
    
    def eval(self):
        """设置评估模式"""
        super().eval()
        self.eval_mode = True
        
    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)
        self.eval_mode = not mode
            
class ImprovedResNetActor(nn.Module):
    """改进的策略网络"""
    def __init__(self, backbone, action_dim=action_dim):
        super().__init__()
        self.backbone = backbone
        backbone_out_dim = backbone.hidden_dim  # 使用Transformer的隐藏维度
        
        # 更深的策略头
        self.fc_policy = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim//2),
            nn.LayerNorm(backbone_out_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_out_dim//2, backbone_out_dim//4),
            nn.LayerNorm(backbone_out_dim//4),
            nn.GELU(),
            nn.Linear(backbone_out_dim//4, action_dim),
            nn.LayerNorm(action_dim)
        )
        
    def forward(self, x, mask=None):
        features = self.backbone(x)
        logits = self.fc_policy(features)
        
        # 动态Pass惩罚系数
        pass_penalty = 1.5 if self.training else 1.0  # 训练时更高惩罚
        # 从状态张量中提取手牌数量（前108维是手牌特征）
        hand_size = int(x[..., :108].sum().item())  # 手牌数量 = 非零特征数量
        if hand_size < 5:  # 终局阶段降低Pass惩罚
            pass_penalty *= max(0.5, 1 - (5 - hand_size)*0.1)
        logits[..., 0] -= pass_penalty
        
        # 应用动作掩码
        if mask is not None:
            logits = logits + (mask.float() - 1) * 1e9
        
        probs = F.softmax(logits, dim=-1)
        return probs, logits

class ImprovedResNetCritic(nn.Module):
    """价值网络"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        backbone_out_dim = backbone.hidden_dim  # 使用Transformer的隐藏维度
        self.fc_value = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim//2),
            nn.LayerNorm(backbone_out_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout
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

# 预分配GPU内存
state_buf = torch.empty((8192, 3049), dtype=torch.float32, device=device)
mask_buf = torch.empty((8192, 456), dtype=torch.float32, device=device)

def bind_to_core(core_id):
    p = psutil.Process(os.getpid())
    p.cpu_affinity([core_id])
    
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

def calculate_team_reward(game):
    """计算队伍获胜奖励
    输入: game对象
    输出: 奖励值(针对玩家0)
    """
    if not game.is_game_over or len(game.ranking) < 2:
        return 0.0
    
    # 获取前两名玩家
    top_two = game.ranking[:2]
    
    # 判断队伍0获胜条件：玩家0和2都在前两名
    team0_win = (0 in top_two) and (2 in top_two)
    
    # 判断队伍1获胜条件：玩家1和3都在前两名
    team1_win = (1 in top_two) and (3 in top_two)
    
    # 队伍获胜奖励
    if team0_win:
        return 2.0  # 队伍0获胜奖励
    elif team1_win:
        return -2.0  # 队伍1获胜惩罚
    else:
        # 混合排名情况
        if 0 in top_two:
            return 1.5  # 玩家0进入前两名奖励
        elif 2 in top_two:
            return 1.0  # 队友玩家2进入前两名奖励
        # 新增：双方队友都在前两名但顺序不同
        elif (0 in top_two and 3 in top_two) or (1 in top_two and 2 in top_two):
            return 0.5  # 混合队伍奖励
    return 0.0

def calculate_improved_reward(entry, player, mask, action_id, hand_size_before, game, ep):
    """改进的奖励计算"""
    reward = 0.0
    progress = (27 - len(player.hand)) / 27  # 游戏进度[0,1]
    
    # 动态基础奖励 - 随游戏进度增加
    base_reward = 0.1 + progress * 0.2
    reward += base_reward
    
    # 炸弹使用奖励调整
    bomb_bonus = {
        'bomb': 0.4 + progress*0.3,
        'straight_bomb': 0.7 + progress*0.4,
        'joker_bomb': 1.2  # 降低天王炸奖励
    }
    hand_size = len(player.hand)
    progress = (27 - hand_size) / 27
    
    # 春天判断 - 第一轮就出完所有牌
    if game.current_round == 1 and hand_size == 0:
        reward += 2.5  # 春天额外奖励
        game.log(f"🎉 春天！玩家 {game.current_player + 1} 在第一轮就出完所有牌！")
    
    # Pass处理优化
        if action_id == 0:
            # 动态Pass惩罚
            base_penalty = 0.5  # 基础惩罚值
            # 根据游戏阶段调整
            if progress < 0.3:  # 早期阶段
                penalty = -base_penalty * 0.8
            elif progress > 0.7:  # 终局阶段
                penalty = -base_penalty * 1.5
            else:
                penalty = -base_penalty
                
            # 自由出牌轮次Pass惩罚加倍
            if game.is_free_turn:
                penalty *= 2.0
                
            reward += penalty
            
            # 连续Pass额外惩罚
            if game.pass_count > 2:
                reward -= 0.1 * game.pass_count
        
    else:
        # 出牌奖励优化
        action_type = entry.get('type', '')
        cards_played = hand_size_before - hand_size
        
        # 基础奖励(考虑剩余牌数)
        base_reward = 0.4 + progress * 0.3
        if hand_size <= len(player.hand) / 2:
            base_reward *= 1.2  # 牌数少于一半时增加奖励
            
        # 牌型奖励优化
        bomb_multiplier = 1.0
        # 关键回合炸弹奖励加成（终局或对抗大牌）
        if progress > 0.7 and len(game.last_play) >= 4:
            bomb_multiplier = 1.5
            
        type_bonus = {
            'single': 0.2 + (1-progress)*0.2,
            'pair': 0.25 + progress*0.1,
            'trio': 0.3 + progress*0.15,
            'bomb': (0.4 + progress*0.2) * bomb_multiplier,  # 普通炸弹
            'straight_bomb': (0.6 + progress*0.3) * bomb_multiplier,  # 顺子炸弹
            'joker_bomb': 1.0 * bomb_multiplier,  # 天王炸
            'sequence': 0.35 + len(game.last_play)*0.04,
            'spring': 5.0  # 春天奖励
        }.get(action_type, 0.0)
        
        # 炸弹使用成本（随游戏进度减少）
        if 'bomb' in action_type:
            bomb_cost = max(0.1, 0.3 * (1 - progress))
            type_bonus -= bomb_cost
        
        # 控制权奖励
        control_bonus = 0.0
        if game.last_player == 0:
            control_bonus = 0.2 + progress * 0.15
            
        # 关键牌奖励
        key_card_bonus = 0.0
        if any(c[0] in ['A', '2'] for c in game.recent_actions[0]):
            key_card_bonus = 0.15 + progress * 0.1
            
        # 增强队友配合奖励
        teammate_bonus = 0.0
        teammate = (game.current_player + 2) % 4
        
        # 基础队友配合
        if game.last_player == teammate:
            teammate_bonus = 0.2
            
        # 接队友牌型额外奖励
        if (game.last_player == teammate and 
            action_type == game.last_play_type and
            cards_played >= len(game.last_play)):
            teammate_bonus += 0.15
            
        # 为队友创造机会奖励
        if (game.last_player != teammate and 
            len(player.hand) < 10 and 
            cards_played == 1 and 
            '2' in game.recent_actions[0][0]):
            teammate_bonus += 0.1
            
        # 策略性额外奖励
        strategy_bonus = 0.0
        # 压制对手奖励
        if game.last_player in [1, 3] and action_type == game.last_play_type:
            strategy_bonus += 0.3
        # 配合队友奖励
        if game.last_player in [0, 2] and action_type == game.last_play_type:
            strategy_bonus += 0.2
        # 获得出牌权奖励
        if game.is_free_turn:
            strategy_bonus += 0.1
            
        reward += (base_reward + type_bonus + control_bonus + 
                  key_card_bonus + teammate_bonus + strategy_bonus)
        
        # 终局优化
        if hand_size <= 3:
            reward += (4 - hand_size) * 0.4
            if hand_size == 0:  # 出完牌额外奖励
                reward += 2.0
                
    return reward


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
                      gamma=0.995, gae_lambda=0.95, device=device, ep=0):
    """优化后的PPO训练函数"""
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.BoolTensor(dones).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)

    # 标准化rewards
    rewards = torch.clamp(rewards, -5.0, 5.0)  # 先裁剪极端值
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 计算优势值和回报
    with torch.no_grad():
        next_values = target_critic(next_states).squeeze(-1)
    values = critic(states).squeeze(-1)
    
    # 使用TD(λ)计算优势
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t].float()) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    # 标准化优势值
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 计算新的动作概率
    probs, logits = actor(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    # 计算KL散度并动态调整clip范围
    kl_div = (old_log_probs - new_log_probs).mean()
    # 动态调整clip范围稳定训练
    if kl_div > 0.015:
        clip_epsilon = 0.1  # 收紧clip范围
    elif kl_div < 0.005:
        clip_epsilon = 0.25  # 放宽clip范围
    else:
        clip_epsilon = 0.15  # 中等范围
    
    # 计算策略损失
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 大幅增强Pass惩罚
    pass_probs = probs[:, 0]
    pass_penalty_factor = 0.8  # 固定高惩罚因子
    pass_penalty = pass_penalty_factor * pass_probs.mean()
    policy_loss += pass_penalty  # 直接应用惩罚
    
    # Value Loss计算优化
    value_pred = critic(states)
    value_targets = returns.unsqueeze(-1)
    value_loss = F.smooth_l1_loss(value_pred, value_targets)
    value_loss = value_loss.clamp(-8.0, 8.0)  # 扩大value loss范围
    
    # 动态熵系数
    if ep < 200:  # 延长初始探索期
        entropy_coef = 0.03  # 提高初始熵系数
    else:
        entropy_coef = max(0.005, 0.03 * (0.9995 ** ((ep-200)//50)))  # 降低熵衰减速度
    
    # 动态调整value loss权重
    value_loss_weight = min(1.0, 0.5 + ep/1000)  # 随训练进程增加value loss权重
    
    # 总损失
    total_loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy
    
    # 检查损失值
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logging.warning(f"Invalid loss detected: {total_loss}")
        return policy_loss.item(), value_loss.item(), entropy.item(), kl_div.item()
    
    # 优化器步骤
    optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪和缩放
    max_grad_norm = 1.0
    grad_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_grad_norm)
    if grad_norm > max_grad_norm:
        for param in backbone.parameters():
            if param.grad is not None:
                param.grad.data.mul_(max_grad_norm / (grad_norm + 1e-6))
    
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    
    optimizer.step()
    
    return policy_loss.item(), value_loss.item(), entropy.item(), kl_div.item()

def run_training(episodes=30000):
    """改进的训练流程 - 添加课程学习和目标网络"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adaptive_params = {
        'min_batch_size': 1024,  # 减小最小batch size
        'max_batch_size': 8192,  # 减小最大batch size
        'batch_growth_interval': 256,  # 减少增加频率
        'current_batch_size': 1024,
        'growth_step': 256  # 减小增长步长
    }
    
    # 改进的课程学习
    def get_curriculum(ep, win_rate):
        if ep < 1000:  # 延长初始训练阶段
            return {'level': (2,4), 'opponent': 'random'}
        elif ep < 3000:
            if win_rate < 0.45:
                return {'level': (3,6), 'opponent': 'random'}
            else:
                return {'level': (4,7), 'opponent': 'rule_based'}
        elif ep < 6000:
            if win_rate < 0.55:
                return {'level': (5,8), 'opponent': 'rule_based'}
            else:
                return {'level': (6,10), 'opponent': 'self'}
        else:
            return {'level': (8,14), 'opponent': 'self'}
    
    # 启用cuDNN benchmark模式
    torch.backends.cudnn.benchmark = True
    
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
    
    # 优化器设置
    optimizer_params = [
        {'params': [p for n,p in actor.named_parameters() 
                if not n.startswith('backbone.')],
         'lr': 3e-5,
         'weight_decay': 1e-5},
        {'params': [p for n,p in critic.named_parameters()
                if not n.startswith('backbone.')],
        'lr': 1.5e-4},  # 降低critic学习率
        {'params': backbone.parameters(),
        'lr': 5e-5}   # 降低backbone学习率
    ]

    optimizer = optim.AdamW(optimizer_params, weight_decay=1e-5)

    # 学习率调度器优化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=150,    # 增加patience
        factor=0.95,     # 降低学习率衰减系数
        min_lr=5e-6,
        verbose=True
    )
    
    '''
    # 使用步数衰减替代Plateau
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=500,  # 每500步衰减一次
        gamma=0.95      # 衰减系数
    )
    '''
    
    num_collectors = 10  # 根据CPU核心数调整
    # 创建线程安全的deque缓冲区列表
    memory_list = []
    for _ in range(num_collectors):
        memory_list.append(ReplayBuffer(capacity=20000))
        
    #memory = ReplayBuffer(capacity=50000)
    writer = SummaryWriter(f'runs/guandan_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    initial_ep = load_checkpoint(device, backbone, actor, critic, optimizer)
    best_reward = float('-inf')
    
    # 初始化训练指标
    policy_loss = float('inf')  # 初始化为一个大值
    value_loss = 0
    entropy = 0
    kl_div = 0
    game_counter = 0  # 牌局计数器
    
    consecutive_pass_rounds = 0
        
    def soft_update(target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    try:
        # 创建数据收集和训练分离的线程
        from threading import Thread
        
        def data_collection_thread(id):
            
            p = psutil.Process()
            cores = list(range(psutil.cpu_count()))
            core_id = id % len(cores)
            p.cpu_affinity([core_id])
            print(f"线程 {id} 绑定到核心 {core_id}")
            
            """独立的数据收集线程"""
            memory = memory_list[id]
            
            for ep in range(initial_ep, initial_ep + episodes):
                thread_id = threading.current_thread().ident
                game_counter = ep - initial_ep + 1
                run_id = datetime.now().strftime("%Y%m%d%H%M%S")
                game_id = f"{run_id}_{game_counter:04d}_{thread_id}"
                
                game = GuandanGame(verbose=False)  # 关闭详细日志
                game.game_id = game_id
                episode_reward = 0
                episode_steps = 0
                pass_penalty_given = False
                continue_rounds = 0                
                            
                while not game.is_game_over and len(game.history) <= 300:
                    # 验证游戏状态
                    if not validate_game_state(game):
                        game.log("⚠️ 游戏状态验证失败，重置状态")
                        game.pass_count = 0
                        game.last_play = []
                        game.is_free_turn = True
                    
                    # 跳过已经出完牌的玩家（即已经在排名中的玩家）    
                    while not game.is_game_over and game.current_player in game.ranking:
                        game.current_player = (game.current_player + 1) % 4
                        game.log(f"玩家 {game.current_player +1}  ERROR")
                        game.is_game_over = len(game.ranking) >= 4
                        game.check_game_over()
                        if game.is_game_over:
                            game.log(f"⚠️ 所有玩家都已出完牌，强制结束游戏 ")

                    continue_rounds = (continue_rounds+1)%2
                    if continue_rounds == 0:
                        consecutive_pass_rounds = consecutive_pass_rounds+game.pass_count
                    else:
                        consecutive_pass_rounds=game.pass_count
                        
                    # 检查连续4次Pass惩罚
                    if consecutive_pass_rounds > 8 and not pass_penalty_given:
                        # 对所有玩家（这里只训练主智能体，reward记录在memory）
                        penalty = -1.0
                        # 仅对主智能体做记录
                        if len(memory) > 0:
                            memory.push(memory.buffer[-1][0], 0, penalty, memory.buffer[-1][0], False, 0.0)
                            episode_reward += penalty
                        # 日志提示
                        game.log(f"所有玩家连续8次Pass，给予所有玩家惩罚！")
                        # 清空pass_count和recent_actions，进入新一轮
                        game.pass_count = 0
                        game.recent_actions = [['None'] for _ in range(4)]
                        pass_penalty_given = True
                    else:
                        pass_penalty_given = False
                        
                    # 当玩家A出牌完后，其余玩家都选择Pass，玩家A重新获得自由出牌权
                    if game.pass_count >= 3:
                        # 记录连续Pass轮次
                        consecutive_pass_rounds = game.pass_count
                        
                        # 重置出牌权给最后出牌的玩家（如果该玩家未出完牌）
                        if game.last_player is not None:
                            # 确保最后出牌的玩家没有出完牌
                            if game.last_player not in game.ranking:
                                game.current_player = game.last_player
                                game.log(f"玩家 {game.last_player + 1} 获得新一轮出牌权（连续{consecutive_pass_rounds}次Pass后） ")
                            else:
                                # 如果最后出牌的玩家已出完牌，则选择下一个未出完牌的玩家
                                next_player = (game.last_player + 1) % 4
                                while next_player in game.ranking:
                                    next_player = (next_player + 1) % 4
                                game.current_player = next_player
                                game.log(f"玩家 {next_player + 1} 获得新一轮出牌权（连续{consecutive_pass_rounds}次Pass后） ")
                        else:
                            # 如果没有最后出牌的玩家，则按顺序找下一个未出完牌的玩家
                            next_player = (game.current_player + 1) % 4
                            while next_player in game.ranking:
                                next_player = (next_player + 1) % 4
                            game.current_player = next_player
                            game.log(f"玩家 {next_player + 1} 获得新一轮出牌权（连续{consecutive_pass_rounds}次Pass后）")

                        # 重置状态
                        game.last_play = []
                        game.pass_count = 0  # 重置Pass计数避免死循环
                        game.recent_actions = [['None'] for _ in range(4)]
                        game.is_free_turn = True
                        game.is_game_over = len(game.ranking) >= 4
                        continue

                    # 玩家0：训练中的PPO智能体（主训练对象）
                    # 玩家1-3：游戏内置的规则型AI对手
                    # 通过current_player轮转机制实现回合制出牌
                    if game.current_player == 0:
                        # 使用缓存机制提升效率
                        if not hasattr(game, 'cached_state') or game.cached_state is None:
                            state = game._get_obs()
                            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                            mask = torch.tensor(game.get_valid_action_mask(game.players[0].hand, M, game.active_level, game.last_play), dtype=torch.float32).unsqueeze(0).to(device)
                            mask = mask.squeeze(0)
                            
                            # 缓存计算结果
                            game.cached_state = state
                            game.cached_state_tensor = state_tensor
                            game.cached_mask = mask
                        else:
                            # 使用缓存
                            state_tensor = game.cached_state_tensor
                            mask = game.cached_mask
                        
                        # 仅在需要时切换模式
                        if actor.training:
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
                        
                        # 不清除缓存，保留用于后续步骤
                        entry = M_id_dict[action_id]
                        player = game.players[0]
                        hand_size_before = len(player.hand)
                        '''
                        state = game._get_obs() #当前游戏状态
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        mask = torch.tensor(game.get_valid_action_mask(game.players[0].hand, M,game.active_level, game.last_play), dtype=torch.float32).unsqueeze(0).to(device) #获取有效动作 掩码处理：无效动作位置为0，有效动作为1，确保智能体只选择合法动作
                        mask = mask.squeeze(0) 
                        actor.eval() #切换actor网络到评估模式（禁用dropout等训练专用层）
                        with torch.no_grad():
                            probs, _ = actor(state_tensor, mask) #输出动作概率分布probs（忽略价值函数输出）
                            temperature = 1.1 #通过温度参数temperature=1.1调整探索程度
                            adj_probs = (probs ** (1/temperature))
                            adj_probs = adj_probs / adj_probs.sum()
                            dist = Categorical(adj_probs)
                            action = dist.sample()
                            log_prob = dist.log_prob(action) #记录动作的对数概率（用于PPO损失计算）
                            action_id = action.item()
                        actor.train() #恢复actor网络到训练模式
                        entry = M_id_dict[action_id]
                        player = game.players[0]
                        hand_size_before = len(player.hand) #记录执行动作前的手牌数量（用于奖励计算）
                        '''
                        
                        # 玩家的动作通过 enumerate_colorful_actions 函数生成可能的组合。
                        # 如果有可选动作，代码随机选择一个动作并更新游戏状态，包括玩家手牌、最近动作记录以及游戏日志。
                        # 如果玩家出完所有牌，代码更新排名并检查游戏是否结束。
                        combos = enumerate_colorful_actions(entry, player.hand, game.active_level) 
                        if combos:
                            chosen_move = random.choice(combos)
                            if not chosen_move:
                                game.log(f"玩家 1 PPO Pass ")
                                game.pass_count += 1
                                game.recent_actions[0] = ['Pass']
                            else:
                                is_free_turn = False
                                game.last_play = chosen_move
                                game.last_player = 0
                                for card in chosen_move:
                                    player.played_cards.append(card)
                                    player.hand.remove(card)
                                game.log(f"玩家 1 PPO 出牌: {' '.join(chosen_move)} , 当前级牌 {game.active_level}")
                                game.recent_actions[0] = list(chosen_move)
                                game.jiefeng = False
                                if not player.hand:
                                    game.log(f"\n🎉 玩家 1 PPO 出完所有牌！\n")
                                    game.ranking.append(0)  # 玩家0的索引直接使用0
                                    game.is_game_over = len(game.ranking) >= 4
                                    game.pass_count = 0
                                    
                                    # 双重确认：确保手牌确实为空
                                    if player.hand:
                                        game.log(f"⚠️ 警告：玩家1PPO手牌非空但被判定为空！手牌: {' '.join(player.hand)} ")
                                    else:
                                        # 轮转玩家并跳出循环
                                        game.current_player = (game.current_player + 1) % 4
                                        continue  # 跳出当前回合循环
                                    
                                game.pass_count = 0
                                # 移除重复的hand检查
                                if game.is_free_turn:
                                    game.is_free_turn = False
                        else:
                            game.log(f"玩家 1  PPO Pass ")
                            game.pass_count += 1
                            game.recent_actions[0] = ['Pass']
                            
                        next_state = game._get_obs()
                        reward = calculate_improved_reward(entry, player, mask, action_id, hand_size_before, game, ep)
                        reward += calculate_team_reward(game)
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
                        # 限制历史记录长度
                        if len(game.history) < 50:
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
                        #print("自由出牌轮，mask:", mask.cpu().numpy())
                    
                    if all(action == ['Pass'] for action in game.recent_actions):
                        game.is_free_turn = True
                        game.last_play = []
                        game.pass_count = 0
                        print("死循环检测，mask:", mask.cpu().numpy())
        
        collectors = []
        for i in range(num_collectors):
            memory = memory_list[i]
            collector = Thread(target=data_collection_thread, daemon=True,kwargs={'id': i})
            collector.start()
        
        # 主训练循环
        collected_episodes = 0
        mennum = 0
        while collected_episodes < episodes:
            mennum += 1
            memory = memory_list[collected_episodes%mennum]
            
            # 批量训练
            if len(memory) >= adaptive_params['current_batch_size']:
                states, actions, rewards, next_states, dones, old_log_probs = memory.sample(
                    adaptive_params['current_batch_size']
                )

                # 从队列获取数据
                collected_episodes += 1

                print(f"len1(memory):{len(memory)} ID:{collected_episodes%mennum}")
                
                # 动态调整batch size
                if collected_episodes % adaptive_params['batch_growth_interval'] == 0:
                    adaptive_params['current_batch_size'] = min(
                        adaptive_params['max_batch_size'],
                        adaptive_params['current_batch_size'] + adaptive_params['growth_step']
                    )
                
                # 使用torch.jit.script加速训练
                with torch.jit.optimized_execution(True):
                    policy_loss, value_loss, entropy, kl_div = train_on_batch_ppo(
                        states, actions, rewards, next_states, dones, old_log_probs,
                        backbone, actor, critic, target_critic, optimizer,
                        gamma=0.99, gae_lambda=0.97, device=device, ep=collected_episodes
                    )
                
                # 记录训练指标
                writer.add_scalar('Training/PolicyLoss', policy_loss, collected_episodes)
                writer.add_scalar('Training/ValueLoss', value_loss, collected_episodes)
                writer.add_scalar('Training/Entropy', entropy, collected_episodes)
                writer.add_scalar('Training/KLDivergence', kl_div, collected_episodes)
                writer.add_scalar('Training/EpisodeSteps', collected_episodes, collected_episodes)
                
                # 减少目标网络更新频率
                if collected_episodes % 50 == 0:                     
                    soft_update(target_backbone, backbone)
                    soft_update(target_critic, critic)
                    
                scheduler.step(policy_loss)
                
                # 定期打印日志    
                if (collected_episodes + 1) % 10 == 0:                
                    logging.info(
                        f"Episode {collected_episodes + 1}: "
                        f"PLoss={policy_loss:.4f}, VLoss={value_loss:.4f}, "
                        f"Entropy={entropy:.4f}, KL={kl_div:.4f}, "
                        f"BatchSize={adaptive_params['current_batch_size']}"
                    )
                
                # 训练后清理已使用的样本
                # 计算实际使用的样本索引
                used_indices = set()
                for i in range(len(memory.buffer)):
                    for j in range(len(states)):
                        if np.array_equal(memory.buffer[i][0], states[j]):
                            used_indices.add(i)
                
                # 移除已使用的样本
                memory.buffer = [item for idx, item in enumerate(memory.buffer) if idx not in used_indices]
                
                # 定期保存检查点
                if (collected_episodes + 1) % 100 == 0:
                    save_checkpoint(backbone, actor, critic, optimizer, collected_episodes + 1) 
            else:
                mennum += 1
                time.sleep(5)

        # 等待所有collector结束
        for collector in collectors:
            collector.join()
                
    except KeyboardInterrupt:
        logging.info("训练被手动中断")
        save_checkpoint(backbone, actor, critic, optimizer, 1000000, model_dir="models/interrupted")
    finally:
        writer.close()

if __name__ == "__main__":
    logging.info("开始训练 掼蛋 PPO 智能体")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"设备: {device}")
    run_training()

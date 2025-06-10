"""
改进的PPO算法实现 - 掼蛋纸牌游戏AI训练
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

# 修改SharedBackbone类
class SharedBackbone(nn.Module):
    def __init__(self, state_dim=3049, hidden_dim=1024):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(state_dim)
        self.eval_mode = False
        
        # 改进的手牌编码器
        self.card_encoder = nn.Sequential(
            nn.Linear(108, 512),
            nn.LayerNorm(512),
            nn.GELU(),  # 使用GELU替代LeakyReLU
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # 改进的历史动作编码器
        self.history_encoder = nn.LSTM(
            input_size=2160,
            hidden_size=512,
            num_layers=3,  # 增加层数
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 改进的场景信息编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(781, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # 多头注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=512+1024+256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 改进的融合层
        self.fusion = nn.Sequential(
            nn.Linear(512+1024+256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 更深的残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate=0.2)
            for _ in range(4)  # 增加残差块数量
        ])

    def forward(self, x):
        x = self.input_bn(x)
        
        # 特征提取
        card_feat = self.card_encoder(x[..., :108])
        
        history_feat, _ = self.history_encoder(x[..., 108:2268].unsqueeze(1))
        history_feat = history_feat.squeeze(1)
        
        context_feat = self.context_encoder(x[..., 2268:])
        
        # 特征融合
        combined_feat = torch.cat([card_feat, history_feat, context_feat], dim=-1)
        
        # 自注意力处理
        attn_out, _ = self.self_attention(
            combined_feat.unsqueeze(1),
            combined_feat.unsqueeze(1),
            combined_feat.unsqueeze(1)
        )
        combined_feat = combined_feat + attn_out.squeeze(1)  # 残差连接
        
        x = self.fusion(combined_feat)
        
        # 残差处理
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
    """改进的策略网络"""
    def __init__(self, backbone, action_dim=action_dim):
        super().__init__()
        self.backbone = backbone
        backbone_out_dim = backbone.res_blocks[-1].fc2.out_features
        
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
        
        # 动态Pass抑制 - 根据训练阶段调整
        if self.training:
            pass_penalty = 1.2  # 训练时中等抑制
        else:
            pass_penalty = 0.8  # 评估时轻微抑制
            
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
        backbone_out_dim = backbone.res_blocks[-1].fc2.out_features
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

def select_action(actor, state, mask, device, is_free_turn, ep):
    """改进的动作选择策略4.0"""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 更平缓的探索衰减曲线
    explore_factor = max(0.15, 0.6 * (0.992 ** (ep // 50)))  # 衰减更慢
    temp = max(0.7, 2.0 - ep/5000)  # 更高的初始温度
    
    # 动态Pass抑制系数
    pass_suppress = 1.8 + (1 - ep/8000)  # 随训练逐渐降低
    
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
        logits += bomb_mask * (0.3 + ep/8000)  # 随训练逐渐重视炸弹
        
        # 改进的探索机制
        if random.random() < explore_factor:
            # 基于Q值的探索
            with torch.no_grad():
                q_values = critic(state_tensor).squeeze()
                valid_q = q_values * mask_tensor[0].float()
                valid_q[0] = -float('inf')  # 排除Pass
                
                # 优先探索高潜力动作
                topk = min(5, (mask_tensor[0] > 0).sum())
                if topk > 0:
                    _, top_actions = valid_q.topk(topk)
                    action = top_actions[random.randint(0, topk-1)]
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
        base_penalty = 0.15 + (1 - progress) * 0.15  # 进一步降低Pass惩罚
        phase_factor = max(0.2, 0.5 - ep/12000)
        penalty = -base_penalty * phase_factor
        
        # 场景相关惩罚调整
        if game.is_free_turn:
            penalty *= 1.2
        elif game.last_play and len(game.last_play) >= 4:
            penalty *= 0.2  # 大幅降低对大牌的Pass惩罚
            
        reward += penalty
        
    else:
        # 出牌奖励优化
        action_type = entry.get('type', '')
        cards_played = hand_size_before - hand_size
        
        # 基础奖励(考虑剩余牌数)
        base_reward = 0.4 + progress * 0.3
        if hand_size <= len(player.hand) / 2:
            base_reward *= 1.2  # 牌数少于一半时增加奖励
            
        # 牌型奖励优化
        type_bonus = {
            'single': 0.2 + (1-progress)*0.2,
            'pair': 0.25 + progress*0.1,
            'trio': 0.3 + progress*0.15,
            'bomb': 0.4 + progress*0.2,  # 普通炸弹
            'straight_bomb': 0.6 + progress*0.3,  # 顺子炸弹
            'joker_bomb': 1.0,  # 天王炸
            'sequence': 0.35 + len(game.last_play)*0.04,
            'spring': 5.0  # 春天奖励
        }.get(action_type, 0.0)
        
        # 控制权奖励
        control_bonus = 0.0
        if game.last_player == 0:
            control_bonus = 0.2 + progress * 0.15
            
        # 关键牌奖励
        key_card_bonus = 0.0
        if any(c[0] in ['A', '2'] for c in game.recent_actions[0]):
            key_card_bonus = 0.15 + progress * 0.1
            
        # 配合队友奖励
        teammate_bonus = 0.0
        teammate = (game.current_player + 2) % 4
        if game.last_player == teammate:
            teammate_bonus = 0.2
            
        reward += (base_reward + type_bonus + control_bonus + 
                  key_card_bonus + teammate_bonus)
        
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
    if kl_div > 0.02:
        clip_epsilon = max(0.1, 0.15)  # 收紧clip范围
    else:
        clip_epsilon = max(0.1, 0.2 * (0.998 ** (ep // 50)))  # 放宽clip范围衰减
    
    # 计算策略损失
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 动态调整Pass惩罚
    pass_probs = probs[:, 0]
    pass_penalty_factor = max(0.02, 0.1 * (0.995 ** ep))  # 降低Pass惩罚
    pass_penalty = pass_penalty_factor * pass_probs.mean()
    policy_loss += pass_penalty
    
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
        'min_batch_size': 512,  # 增大初始batch size
        'max_batch_size': 8192,  # 增大最大batch size
        'batch_growth_interval': 64,  # 更频繁调整
        'current_batch_size': 512,
        'growth_step': 128  # 增大增长步长
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
    
    memory = ReplayBuffer(capacity=10000)
    writer = SummaryWriter(f'runs/guandan_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    initial_ep = load_checkpoint(device, backbone, actor, critic, optimizer)
    best_reward = float('-inf')
    
    # 初始化训练指标
    policy_loss = float('inf')  # 初始化为一个大值
    value_loss = 0
    entropy = 0
    kl_div = 0
    
    def soft_update(target, source, tau=0.005):
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

                # 玩家0：训练中的PPO智能体（主训练对象）
                # 玩家1-3：游戏内置的规则型AI对手
                # 通过current_player轮转机制实现回合制出牌
                if game.current_player == 0:
                    state = game._get_obs() #玩家状态
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    mask = torch.tensor(game.get_valid_action_mask(game.players[0].hand, M,
                                                 game.active_level, game.last_play),
                        dtype=torch.float32).unsqueeze(0).to(device) #获取有效动作
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
                    # 玩家的动作通过 enumerate_colorful_actions 函数生成可能的组合。
                    # 如果有可选动作，代码随机选择一个动作并更新游戏状态，包括玩家手牌、最近动作记录以及游戏日志。
                    # 如果玩家出完所有牌，代码更新排名并检查游戏是否结束。
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
        
                # 动态调整batch size
                if ep % adaptive_params['batch_growth_interval'] == 0:
                    adaptive_params['current_batch_size'] = min(
                        adaptive_params['max_batch_size'],
                        adaptive_params['current_batch_size'] + 32
                    )
                
                if len(memory) >= adaptive_params['current_batch_size']:
                    states, actions, rewards, next_states, dones, old_log_probs = memory.sample(
                        adaptive_params['current_batch_size']
                    )
                    # 添加训练步骤并接收返回值
                    policy_loss, value_loss, entropy, kl_div = train_on_batch_ppo(
                        states, actions, rewards, next_states, dones, old_log_probs,
                        backbone, actor, critic, target_critic, optimizer,
                        gamma=0.99, gae_lambda=0.97, device=device, ep=ep
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
                    
            if (ep + 1) % 10 == 0:
                # 添加梯度范数监控
                grad_norms = []
                for param in backbone.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                avg_grad_norm = sum(grad_norms)/len(grad_norms) if grad_norms else 0
                
                # 添加学习率监控
                lrs = [group['lr'] for group in optimizer.param_groups]
                
                logging.info(
                    f"Episode {ep + 1}: "
                    f"PLoss={policy_loss:.4f}, VLoss={value_loss:.4f}, "
                    f"Entropy={entropy:.4f}, KL={kl_div:.4f}, "
                    f"Reward={episode_reward:.2f}, Steps={episode_steps}, "
                    f"BatchSize={adaptive_params['current_batch_size']}"
                )
                    
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_checkpoint(backbone, actor, critic, optimizer, ep + 1,model_dir="models/best")
                
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

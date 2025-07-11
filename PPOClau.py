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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")
action_dim = 1024

class ReplayBuffer:
    """存储训练样本的循环缓冲区"""
    def __init__(self, capacity=2050):
        self.buffer = []  # 存储样本的列表
        self.capacity = capacity # 最大容量
        self.position = 0
        self._lock = threading.RLock()  # 使用可重入锁
        
    def push(self, state, action, reward, next_state, done, log_prob):
        """添加新样本"""
        # 确保数据有效
        if state is None or next_state is None or log_prob is None:
            return
        
        # 使用线程锁确保线程安全
        with self._lock:
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
        with self._lock:
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
    def __init__(self, state_dim=1430, hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 改进的输入层
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 移除LSTM层，改用残差块
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 初始特征提取
        x = self.input_layer(x)
        
        # 残差连接
        residual = x
        x = self.res_block1(x)
        x = x + residual  # 第一次残差连接
        
        residual = x
        x = self.res_block2(x)
        x = x + residual  # 第二次残差连接
        
        return x

class ImprovedResNetActor(nn.Module):
    """LSTM策略网络"""
    def __init__(self, backbone, action_dim=action_dim):
        super().__init__()
        self.backbone = backbone
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(backbone.hidden_dim, backbone.hidden_dim//2),
            nn.LayerNorm(backbone.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(backbone.hidden_dim//2, action_dim)
        )
        
    def forward(self, x, mask=None):
        features = self.backbone(x)
        logits = self.policy_head(features)
        
        # 应用动作掩码
        if mask is not None:
            logits = logits + (mask.float() - 1) * 1e9
        
        probs = F.softmax(logits, dim=-1)
        return probs, logits

class ImprovedResNetCritic(nn.Module):
    """LSTM价值网络"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(backbone.hidden_dim, backbone.hidden_dim//2),
            nn.LayerNorm(backbone.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(backbone.hidden_dim//2, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        value = self.value_head(features)
        return value

# 预分配GPU内存
state_buf = torch.empty((8192, 1430), dtype=torch.float32, device=device)
mask_buf = torch.empty((8192, 456), dtype=torch.float32, device=device)

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
        return 4.5  if game.current_player in [0,2] else 6  # 根据位置调整
    elif team1_win:
        return -4.5  if game.current_player in [1,3] else -6 
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

def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    last_gae = 0
    
    # 反向计算GAE
    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = next_values[t]
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    
    # 更稳定的标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
    # 确保所有张量都在同一设备上
    model_device = next(critic.parameters()).device
    states = states.to(model_device)
    next_states = next_states.to(model_device)
    actions = actions.to(model_device)
    rewards = rewards.to(model_device)
    dones = dones.to(model_device)
    old_log_probs = old_log_probs.to(model_device)
    
    # 标准化rewards
    #rewards = torch.clamp(rewards, -10.0, 18.0)  # 先裁剪极端值
    #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 计算优势值和回报
    with torch.no_grad():
        next_values = target_critic(next_states).squeeze(-1)
    values = critic(states).squeeze(-1)
    
    # === 计算 GAE 和 Returns ===
    advantages, returns = compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95)
        
    # 标准化优势值
    # 更稳定的优势值标准化
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=True) + 1e-7  # 使用无偏估计
    advantages = (advantages - adv_mean) / adv_std

    # 计算新的动作概率  PPO 核心
    probs, logits = actor(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean() # 熵正则项
    
    # 计算KL散度并动态调整clip范围
    kl_div = (old_log_probs - new_log_probs).mean()
    # 更严格的KL控制 - 防止策略更新过大
    if kl_div > 0.01:  # 降低阈值
        clip_epsilon = 0.05  # 更紧的clip范围
    elif kl_div < 0.002:
        clip_epsilon = 0.3   # 放宽clip范围
    else:
        clip_epsilon = 0.15
    
    # 计算策略损失（带clip）
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 使用更稳定的clip实现
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # Value Loss计算优化
    value_pred = critic(states)
    value_targets = returns.unsqueeze(-1)
    
    # 使用Huber损失替代smooth_l1_loss，更稳定
    value_loss = F.huber_loss(value_pred, value_targets, reduction='mean', delta=1.0)
    
    # 增强后期探索 - 解决Entropy稳定问题
    entropy_coef = 0.01  # 固定熵系数，简化设置
    
    # 价值损失权重固定为0.5
    value_loss_weight = 0.5
    
    # 移除KL惩罚项
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

def run_training(episodes=300000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adaptive_params = {
        'min_batch_size': 1024,  # 减小最小batch size
        'max_batch_size': 2048,  # 减小最大batch size
        'batch_growth_interval': 256,  # 减少增加频率
        'current_batch_size': 1024,
        'growth_step': 30  # 减小增长步长
    }
    
    # 改进的课程学习
    def get_curriculum(ep, win_rate):
        # 更平缓的过渡
        if ep < 5000:  # 延长随机对手阶段
            return {'level': (3,5), 'opponent': 'random'}
        elif ep < 10000:  # 延长规则对手阶段
            return {'level': (5,8), 'opponent': 'rule_based'}
    
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
    # 优化器设置 - 避免参数重复分组
    optimizer_params = [
        {'params': backbone.parameters(), 'lr': 5e-5},
        {'params': [p for n, p in actor.named_parameters() if not n.startswith('backbone.')], 'lr': 3e-5},
        {'params': [p for n, p in critic.named_parameters() if not n.startswith('backbone.')], 'lr': 1e-4}
    ]

    optimizer = optim.Adam(optimizer_params)

    # 学习率调度器优化
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=500,  # 半周期长度
        eta_min=1e-6  # 最小学习率
    )
    
    '''
    # 使用步数衰减替代Plateau
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=500,  # 每500步衰减一次
        gamma=0.95      # 衰减系数
    )
    '''
    
    num_collectors = 10 # 根据CPU核心数调整
    # 创建线程安全的deque缓冲区列表
    memory_list = []
    for _ in range(num_collectors):
        memory_list.append(ReplayBuffer(capacity=2050))
        
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
        
    def soft_update(target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    try:
        # 创建数据收集和训练分离的线程
        from threading import Thread
        
        def data_collection_thread(id):
            # 为每个线程创建独立的CUDA流
            stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
            
            # 创建本地模型副本并移到GPU
            with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
                local_actor = deepcopy(actor).to(device)
                local_actor.eval()
                local_critic = deepcopy(critic).to(device)
                local_critic.eval()
            
            local_step = 0

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
                
                game = GuandanGame(verbose=True)  # 关闭详细日志
                game.log(f"\n\n🎮 游戏开始！当前级牌：{[game.active_level]}")
                game.game_id = game_id
                episode_reward = 0
                episode_steps = 0
                pass_penalty_given = False
                continue_rounds = 0                
                            
                while not game.is_game_over and len(game.history) <= 200:
                    # 验证游戏状态
                    if not validate_game_state(game):
                        game.log("⚠️ 游戏状态验证失败，重置状态")
                        game.pass_count = 0
                        game.last_play = []
                        game.is_free_turn = True
                    
                    # 跳过已经出完牌的玩家（即已经在排名中的玩家）    
                    while not game.is_game_over and game.current_player in game.ranking:
                        game.current_player = (game.current_player + 1) % 4
                        game.log(f"玩家 {game.current_player +1}  ERROR ")
                        game.is_game_over = len(game.ranking) >= 4
                        game.check_game_over()
                        if game.is_game_over:
                            game.log(f"⚠️ 所有玩家都已出完牌，强制结束游戏 ")
                        
                    # 检查连续4次Pass惩罚
                    if game.pass_count > 4 and not pass_penalty_given:
                        # 对所有玩家（这里只训练主智能体，reward记录在memory）
                        penalty = -1.0
                        # 仅对主智能体做记录
                        if len(memory) > 0:
                            memory.push(memory.buffer[-1][0], 0, penalty, memory.buffer[-1][0], False, 0.0)
                            episode_reward += penalty
                        # 日志提示
                        game.log(f"所有玩家连续4次Pass，给予所有玩家惩罚！")
                        
                    # 当玩家A出牌完后，其余玩家都选择Pass，玩家A重新获得自由出牌权
                    if game.pass_count >= 3:
                        # 重置出牌权给最后出牌的玩家（如果该玩家未出完牌）
                        if game.last_player is not None:
                            # 确保最后出牌的玩家没有出完牌
                            if game.last_player not in game.ranking:
                                game.current_player = game.last_player
                                game.log(f"玩家 {game.last_player + 1} 获得新一轮出牌权（连续{game.pass_count}次Pass后） ")
                            else:
                                # 如果最后出牌的玩家已出完牌，则选择下一个未出完牌的玩家
                                next_player = (game.last_player + 1) % 4
                                while next_player in game.ranking:
                                    next_player = (next_player + 1) % 4
                                game.current_player = next_player
                                game.log(f"玩家 {next_player + 1} 获得新一轮出牌权（连续{game.pass_count}次Pass后） ")
                        else:
                            # 如果没有最后出牌的玩家，则按顺序找下一个未出完牌的玩家
                            next_player = (game.current_player + 1) % 4
                            while next_player in game.ranking:
                                next_player = (next_player + 1) % 4
                            game.current_player = next_player
                            game.log(f"玩家 {next_player + 1} 获得新一轮出牌权（连续{game.pass_count}次Pass后）")

                        # 重置状态
                        game.pass_count = 0  # 重置Pass计数避免死循环
                        game.recent_actions = [['None'] for _ in range(4)]
                        game.is_free_turn = True
                        game.last_play = []
                        game.is_game_over = len(game.ranking) >= 4
                        continue

                    # 玩家0：训练中的PPO智能体（主训练对象）
                    # 玩家1-3：游戏内置的规则型AI对手
                    # 通过current_player轮转机制实现回合制出牌
                    if game.current_player == 0:

                        combos = []
                        chosen_move_temp = 0
                        player = game.players[0]
                        action_id = 0
                        
                        # 在CUDA流中进行推理
                        with torch.cuda.stream(stream) if stream else contextlib.nullcontext(), torch.no_grad():
                            # 0. 直接计算状态和掩码（移除缓存机制）
                            state = game._get_obs()
                        
                            # 创建状态张量并移到GPU
                            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                            
                            # 创建合法动作掩码
                            mask_tensor = torch.zeros(action_dim, dtype=torch.float32, device=device)
                            
                            # 1. 获取合法动作列表
                            combos = game.get_possible_moves(player.hand)
                            
                            # 创建临时的action_id_val与combos对照表
                            action_id_to_combo_map = {}
                            if combos:
                                # 循环计算每个combo对应的action_id
                                for combo in combos:
                                    action_id_val= game.rules.cards_to_mod_id(combo[0])
                                    
                                    #action_id_val = game.rules.PLAY_MAPPING.get(combo[1], -1)
                                    #if action_id_val != -1:
                                    #    action_id_val = action_id_val*10000 + combo[2]
                                    #    action_id_val = action_id_val % 1023
                                    
                                    # 添加到对照表
                                    if 0 <= action_id_val < action_dim:
                                        # 使用combo[0]作为键值（出牌组合）
                                        action_id_to_combo_map[action_id_val] = combo[0]
                                        mask_tensor[action_id_val] = 1.0
                            else:
                                # 没有合法动作时只允许Pass
                                mask_tensor[0] = 1.0
                                action_id_to_combo_map[0] = []  # Pass动作
                            
                            # 2. 模型推理获取动作概率
                            probs, logits = local_actor(state_tensor, mask_tensor)
                            
                            # 3. 价值引导的探索策略
                            state_value = local_critic(state_tensor).item()
                            value_based_temp = max(0.5, 1.5 - state_value * 0.1)  # 低价值状态增加探索
                            
                            # 4. 结合不确定性调整温度参数
                            uncertainty = 1.0 - probs.max(dim=1)[0].mean().item()
                            temperature = max(0.5, value_based_temp + 2.0 * uncertainty)
                            
                            # 5. 调整概率分布
                            adj_probs = probs ** (1/temperature)
                            
                            # 6. 数值稳定性检查
                            prob_sum = adj_probs.sum()
                            if prob_sum == 0 or torch.isnan(prob_sum):
                                # 回退到有效动作掩码
                                adj_probs = mask_tensor.clone().float()
                                if adj_probs.sum() == 0:
                                    # 如果没有有效动作，默认选择Pass
                                    adj_probs[0] = 1.0
                            else:
                                adj_probs = adj_probs / prob_sum
                            
                            # 7. 再次检查概率分布有效性
                            if torch.isnan(adj_probs).any():
                                # 创建安全分布
                                adj_probs = torch.ones_like(probs) / probs.size(-1)
                            
                            # 8. 从调整后的分布中采样动作
                            dist = Categorical(adj_probs)
                            action = dist.sample()
                            log_prob = dist.log_prob(action)
                            action_id = action.item()
                        
                        # 9. 更新本地步数计数器
                        local_step += 1
                        
                        # 10. 定期同步模型参数
                        if local_step % 5 == 0:
                            # 同步参数
                            local_actor = deepcopy(actor).to(device)
                            local_actor.eval()
                            local_critic = deepcopy(critic).to(device)
                            local_critic.eval()
                        
                        # 如果有可选动作，代码随机选择一个动作并更新游戏状态，包括玩家手牌、最近动作记录以及游戏日志。
                        # 如果玩家出完所有牌，代码更新排名并检查游戏是否结束。

                        if action_id in action_id_to_combo_map:
                            chosen_move = action_id_to_combo_map[action_id]

                            if not chosen_move:
                                game.log(f"玩家 1 PPO Pass ")
                                game.pass_count += 1
                                game.recent_actions[0] = ['Pass']
                            else:
                                game.is_free_turn = False
                                game.last_play = chosen_move
                                game.last_player = 0
                                for card in chosen_move:
                                    player.played_cards.append(card)
                                    try:
                                        player.hand.remove(card)
                                    except Exception as e:
                                        game.log(f"⚠️ 玩家0出牌时发生异常: {str(e)}")
                                game.log(f"玩家 1 PPO 出牌: {' '.join(chosen_move)} , 当前级牌 {game.active_level}")
                                game.recent_actions[0] = list(chosen_move)
                                game.jiefeng = False
                                if not player.hand:
                                    game.log(f"🎉 玩家 1 PPO 出完所有牌！\n")
                                    game.ranking.append(0)  # 玩家0的索引直接使用0
                                    game.is_game_over = len(game.ranking) >= 4

                                game.pass_count = 0
                                # 移除重复的hand检查
                                if game.is_free_turn:
                                    game.is_free_turn = False
                        else:
                            game.log(f"玩家 1  PPO Pass ")
                            game.pass_count += 1
                            game.recent_actions[0] = ['Pass']
                            
                        next_state = game._get_obs()

                        game_over = game.check_game_over()
                        if not game_over:
                            next_player = (game.current_player + 1) % 4
                            while next_player in game.ranking :
                                next_player = (next_player + 1) % 4
                                game.current_player = next_player
                                            
                        # 计算即时奖励（不含团队奖励）
                        reward = game.compute_reward()

                        # 添加缓冲区大小检查
                        if len(memory) < memory.capacity:
                            memory.push(state, action_id, reward, next_state, game.is_game_over, log_prob.item())
                        else:
                            # 缓冲区满时移除最旧的经验
                            memory.buffer.pop(0)
                            memory.push(state, action_id, reward, next_state, game.is_game_over, log_prob.item())
                            
                        player.last_played_cards = game.recent_actions[0]
                        game.current_player = (game.current_player + 1) % 4 
                        episode_steps += 1
                    else:
                        # 添加AI玩家结束检查
                        current_ai_player = game.players[game.current_player]
                        game.ai_play(current_ai_player)
                        
                        # 检查AI玩家是否出完牌（使用玩家索引代替seat属性）
                        if game.is_game_over:
                                break  # 立即结束游戏循环
                            
                    round_history = []
                    if game.current_player == 0 and any(action != ['None'] for action in game.recent_actions):
                        round_history = [action.copy() for action in game.recent_actions]
                        game.history.append(round_history)
                        game.log(f">>> {len(game.history)}轮 <<<")

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
            memory = memory_list[collected_episodes%num_collectors]
            
            # 批量训练
            if len(memory) >= adaptive_params['current_batch_size']:
                states, actions, rewards, next_states, dones, old_log_probs = memory.sample(adaptive_params['current_batch_size'])

                # 将 numpy 数据转为 tensor，并送入 GPU
                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).long().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                dones = torch.from_numpy(dones).float().to(device)
                old_log_probs = torch.from_numpy(old_log_probs).float().to(device)
                
                # 确保模型在同一个设备上
                backbone = backbone.to(device)
                actor = actor.to(device)
                critic = critic.to(device)
                target_critic = target_critic.to(device)

                # 从队列获取数据
                collected_episodes += 1

                print(f"ID:{collected_episodes%num_collectors} LEN:{len(memory)} TIME:{datetime.now().strftime('%Y%m%d-%H:%M:%S')} Mem:{len(memory)} NO.{collected_episodes%mennum} ")
                
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
                #used_indices = set()
                #for i in range(len(memory.buffer)):
                #    for j in range(len(states)):
                #        if np.array_equal(memory.buffer[i][0], states[j]):
                #            used_indices.add(i)
                
                # 移除已使用的样本
                #memory.buffer = [item for idx, item in enumerate(memory.buffer) if idx not in used_indices]
                memory.buffer = []
                
                # 定期保存检查点
                if (collected_episodes + 1) % 100 == 0:
                    save_checkpoint(backbone, actor, critic, optimizer, collected_episodes + 1) 
            else:
                mennum += 1
                print(f"牌局不够:{len(memory)} NO.{collected_episodes%mennum} TIME:{datetime.now().strftime('%Y%m%d-%H:%M:%S')}")
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
    logging.info("开始训练 掼蛋 PPO 智能体 ")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"设备: {device}")
    run_training()
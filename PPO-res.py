"""
掼蛋多智能体PPO强化学习实现
核心改进：
1. 多智能体训练框架
2. 增强的状态表示
3. 改进的奖励机制
4. 异步训练支持
"""

import os
import math
import numpy as np
import torch
import time
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import json
import random
from pathlib import Path
from collections import deque
import multiprocessing as mp
from typing import List, Dict, Optional
from dataclasses import dataclass
from guandan_env import GuandanGame
import time
from datetime import datetime
    
# === 1. 配置与常量 ===
@dataclass
class Config:
    def __init__(self):
        # 从 json 文件加载动作空间大小
        with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
            self.M = json.load(f)  # 修改为实例变量
        self.M_id_dict = {a['id']: a for a in self.M}  # 修改为实例变量
        self.action_dim = len(self.M)  # 设置动作空间维度
        
        # 添加 PPO 相关参数
        self.ppo_epochs = 10  # PPO 更新轮数
        self.value_clip = 0.2  # 价值函数裁剪范围
        self.max_grad_norm = 0.5  # 梯度裁剪范围
        
        # 其他参数保持不变
        self.state_dim = 4096
        self.hidden_dim = 1024
        self.num_layers = 4
        self.batch_size = 512
        self.num_epochs = 3000
        self.update_steps = 2048
        self.num_envs = 4
        self.gamma = 0.98
        self.gae_lambda = 0.92
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.entropy_decay = 0.995
        self.min_entropy_coef = 0.001
        self.lr = 3e-4
        self.weight_decay = 1e-5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. 增强的状态编码 ===
class StateEncoder:
    """增强的状态表示编码器"""
    def __init__(self, config: Config):
        self.config = config
        self.M = config.M
        self.M_id_dict = config.M_id_dict

    def _get_action_id(self, action: List[str]) -> Optional[int]:
        """获取动作ID"""
        if not action or action[0] == 'Pass':
            return 0
        
        try:
            for action_entry in self.M:
                if set(action_entry['cards']) == set(action):
                    return action_entry['id']
        except:
            pass
        return None
    
    def encode_hand(self, cards: List[str]) -> np.ndarray:
        """编码手牌"""
        # 创建一个54维的one-hot向量(52张普通牌 + 2张王)
        encoding = np.zeros(54, dtype=np.float32)
        
        # 牌面到索引的映射
        ranks = '34567890JQKA2'  # 0代表10
        suits = 'CDHS'  # 梅花、方块、红桃、黑桃
        
        for card in cards:
            if card == 'BJ':  # 大王
                encoding[52] = 1
            elif card == 'RJ':  # 小王
                encoding[53] = 1
            else:
                # 普通牌的编码: suit * 13 + rank
                rank = card[0] if card[0] != '1' else '0'  # 10特殊处理
                suit = card[-1]
                rank_idx = ranks.index(rank)
                suit_idx = suits.index(suit)
                card_idx = suit_idx * 13 + rank_idx
                encoding[card_idx] = 1
                
        return encoding
    
    def encode_history(self, game: GuandanGame, max_rounds: int = 3) -> np.ndarray:
        """编码最近几轮的出牌历史"""
        # 创建历史动作的编码矩阵
        history_size = max_rounds * 4 * self.config.action_dim
        history_encoding = np.zeros(history_size, dtype=np.float32)
        
        # 获取最近max_rounds轮的历史
        recent_history = list(game.history)[-max_rounds:]
        
        for round_idx, round_actions in enumerate(recent_history):
            for player_idx, action in enumerate(round_actions):
                if action and action[0] != 'None' and action[0] != 'Pass':
                    try:
                        # 将action转换为action_id
                        action_id = self._get_action_id(action)
                        if action_id is not None:
                            # 计算在扁平化数组中的位置
                            offset = (round_idx * 4 + player_idx) * self.config.action_dim
                            history_encoding[offset + action_id] = 1
                    except:
                        pass
                        
        return history_encoding

    def encode_relationships(self, game: GuandanGame, player_id: int) -> np.ndarray:
        """编码玩家间关系（队友/对手）"""
        relationships = np.zeros(4, dtype=np.float32)
        teammate_id = (player_id + 2) % 4
        relationships[teammate_id] = 1  # 标记队友
        return relationships

    def encode_game_state(self, game: GuandanGame, player_id: int) -> np.ndarray:
        """完整的状态编码"""
        # 1. 基础状态 (修改这里，不传入 player 参数)
        base_state = game._get_obs()  # 移除 player 参数
        
        # 2. 手牌编码
        hand = self.encode_hand(game.players[player_id].hand)
        
        # 3. 历史动作
        history = self.encode_history(game)
        
        # 4. 玩家关系
        relationships = self.encode_relationships(game, player_id)
        
        # 5. 玩家特定信息
        player_info = self._encode_player_info(game, player_id)
        
        # 6. 其他游戏信息
        game_info = np.array([
            game.active_level / 13,  # 归一化的级牌等级
            len(game.ranking) / 4,   # 归一化的已完成玩家数
            game.pass_count / 3,     # 归一化的pass计数
            1 if game.is_free_turn else 0,  # 自由出牌标志
        ], dtype=np.float32)
        
        # 合并所有特征
        state = np.concatenate([
            base_state,
            hand,
            history,
            relationships,
            player_info,
            game_info
        ])
        
        return state
    
    def _encode_player_info(self, game: GuandanGame, player_id: int) -> np.ndarray:
        """编码玩家特定信息"""
        player = game.players[player_id]
        
        # 获取各玩家手牌数量
        hand_counts = np.array([len(p.hand) for p in game.players]) / 27.0  # 归一化
        
        # 获取最近出牌信息
        recent_plays = np.zeros(4)
        for i, actions in enumerate(game.recent_actions):
            if actions and actions[0] != 'None':
                recent_plays[i] = 1
                
        # 玩家位置信息（相对于当前玩家的位置）
        positions = np.zeros(4)
        positions[player_id] = 1
        
        # 合并信息
        info = np.concatenate([
            hand_counts,
            recent_plays,
            positions
        ])
        
        return info
    
    
    @staticmethod
    def _card_to_index(card: str) -> int:
        """将牌面转换为索引"""
        if card == 'BJ':
            return 52
        elif card == 'RJ':
            return 53
        else:
            ranks = '34567890JQKA2'
            suits = 'CDHS'
            rank = card[0] if card[0] != '1' else '0'
            suit = card[-1]
            rank_idx = ranks.index(rank)
            suit_idx = suits.index(suit)
            return suit_idx * 13 + rank_idx

# === 3. 改进的神经网络架构 ===
class TransformerBlock(nn.Module):
    """注意力机制处理玩家间关系"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        x = x + self.attention(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # FFN
        x = x + self.mlp(self.ln2(x))
        return x

class ImprovedActor(nn.Module):
    """改进的Actor网络"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.action_dim)
        )
        
    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.feature_net(state)
        
        # Transformer处理
        x = x.unsqueeze(0)  # [1, B, D]
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(0)
        
        # 输出策略分布
        logits = self.policy_head(x)
        
        # 应用动作掩码
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
            
        return F.softmax(logits, dim=-1)

class ImprovedCritic(nn.Module):
    """改进的Critic网络"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 特征提取(同Actor)
        self.feature_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # 值函数头
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.feature_net(state)
        
        # Transformer处理
        x = x.unsqueeze(0)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(0)
        
        return self.value_head(x)

# === 4. 改进的奖励设计 ===
class RewardShaper:
    def __init__(self, config: Config):
        self.config = config
        self.M = config.M
        self.M_id_dict = config.M_id_dict
        self.entropy_coef = config.entropy_coef  # 添加熵系数
        
    def _compute_action_value(self, action_id: int, game: GuandanGame) -> float:
        if action_id == 0:
            return 0.0
            
        action = self.M_id_dict[action_id]  # 使用实例变量
        base_value = len(action['cards'])
        
        pattern_values = {
            'single': 1.0,
            'pair': 1.2,
            'trio': 1.5,
            'bomb': 2.0,
            'straight': 1.8,
            'sequence': 1.6
        }
        
        pattern = action['pattern']
        value_multiplier = pattern_values.get(pattern, 1.0)
        
        return base_value * value_multiplier
        
    def compute_immediate_reward(
        self,
        game: GuandanGame,
        player_id: int,
        action_id: int,
        action_mask: np.ndarray,
        cards_played: int
    ) -> float:
        """计算即时奖励"""
        reward = 0.0
        
        # 1. 基础动作奖励
        if action_mask[action_id]:  # 合法动作
            reward += self._compute_action_value(action_id, game)
            reward += 0.05 * cards_played  # 出牌数量奖励
            
            if self._is_bomb(action_id):
                reward += self._compute_bomb_reward(action_id, game)
                
        else:  # 非法动作惩罚
            reward -= 1.0
            
        # 2. 策略奖励
        reward += self._compute_strategy_reward(game, player_id, action_id)
        
        # 3. 局势奖励
        reward += self._compute_situation_reward(game, player_id)
        
        return reward
    
    def _is_bomb(self, action_id: int) -> bool:
        """判断是否为炸弹"""
        if action_id == 0:
            return False
        action = self.M_id_dict[action_id]  # 使用实例变量
        return action['pattern'] == 'bomb'
    
    def compute_team_reward(
        self,
        game: GuandanGame,
        player_id: int,
        upgrade_amount: int
    ) -> float:
        """计算团队奖励"""
        team_reward = 0.0
        
        # 基础升级奖励
        if upgrade_amount > 0:
            team_reward += min(5.0, upgrade_amount * 1.5)
            
        # 胜负奖励
        if len(game.ranking) == 4:  # 游戏结束
            team = 0 if player_id in [0, 2] else 1
            team_players = [0, 2] if team == 0 else [1, 3]
            
            # 计算双方最佳完成名次
            team_best = min(game.ranking.index(p) for p in team_players if p in game.ranking)
            opponent_best = min(game.ranking.index(p) for p in [1, 3] if p in game.ranking)
            
            if team_best < opponent_best:
                team_reward += 3.0  # 获胜奖励
            
        return team_reward
            
    def _compute_bomb_reward(self, action_id: int, game: GuandanGame) -> float:
        """计算炸弹奖励"""
        action = self.M_id_dict[action_id]
        bomb_size = len(action['cards'])
        # 炸弹价值随大小增加
        return 0.5 * bomb_size
        
    def _compute_lead_reward(self, game: GuandanGame, player_id: int, action_id: int) -> float:
        """计算引导性出牌的奖励"""
        if action_id == 0:
            return 0.0
            
        reward = 0.0
        action = self.M_id_dict[action_id]
        
        # 1. 主动出牌时选择合适的牌型
        if game.is_free_turn:
            # 优先出小牌
            if action['logic_point'] <= 7:  # 7以下的牌
                reward += 0.2
            # 鼓励出顺子/连对等组合牌
            if action['pattern'] in ['straight', 'sequence']:
                reward += 0.3
                
        # 2. 为队友创造机会
        teammate_id = (player_id + 2) % 4
        teammate_hand_size = len(game.players[teammate_id].hand)
        if teammate_hand_size <= 5:  # 队友快出完时
            if action['logic_point'] >= 10:  # 出大牌
                reward += 0.4
                
        return reward

    def _compute_strategy_reward(
        self,
        game: GuandanGame,
        player_id: int,
        action_id: int
    ) -> float:
        """计算策略性奖励"""
        reward = 0.0
        
        # 1. 配合队友
        teammate_id = (player_id + 2) % 4
        if game.last_player == teammate_id:
            # 接队友牌的奖励
            reward += 0.2
            
        # 2. 干扰对手
        opponents = [(player_id + 1) % 4, (player_id + 3) % 4]
        if game.last_player in opponents:
            # 压制对手牌的奖励
            reward += 0.3
            
        # 3. 节奏控制
        if game.is_free_turn:
            # 自由出牌时的策略奖励
            reward += self._compute_lead_reward(game, player_id, action_id)
            
        return reward
    
    def _compute_situation_reward(self, game: GuandanGame, player_id: int) -> float:
        """计算局势奖励"""
        reward = 0.0
        
        # 1. 手牌数量
        hand_size = len(game.players[player_id].hand)
        if hand_size <= 5:
            reward += 0.1 * (5 - hand_size)  # 少牌奖励
            
        # 2. 相对位置
        if len(game.ranking) > 0:
            if player_id not in game.ranking:  # 未出完
                reward -= 0.1 * len(game.ranking)  # 落后惩罚
                
        # 3. 团队形势
        teammate_id = (player_id + 2) % 4
        if teammate_id in game.ranking:  # 队友已出完
            reward += 0.2  # 鼓励快速出完
            
        return reward
    
# === 5. 异步训练支持 ===
class AsyncRunner:
    """异步环境运行器"""
    def __init__(self, config: Config, actor: ImprovedActor, reward_shaper: RewardShaper):
        self.config = config
        self.actor = actor
        self.state_encoder = StateEncoder(config)
        self.reward_shaper = reward_shaper
        
    def run_episode(self, pid: int) -> Dict:
        """运行单个环境的一个episode"""
        MAX_STEPS = 1000  # 添加最大步数限制
        step_count = 0
    
        game = GuandanGame(verbose=False)
        memory = []
        
        while not game.is_game_over and step_count < MAX_STEPS:
            step_count += 1
            current_player = game.current_player
            
            # 获取增强的状态表示 (不需要修改 _get_obs)
            state = self.state_encoder.encode_game_state(game, current_player)
            
            # 获取动作掩码
            mask = game.get_valid_action_mask(
                game.players[current_player].hand,
                self.config.M,  # 使用 config 中的变量
                game.active_level,
                game.last_play
            )
            
            # 动作采样与执行
            action_id = self._sample_action(state, mask)
            cards_played = self._execute_action(game, action_id)
            
            # 计算奖励
            reward = self.reward_shaper.compute_immediate_reward(
                game, current_player, action_id, mask, cards_played
            )
            
            # 记录经验
            if current_player == pid:
                memory.append({
                    "state": state,
                    "action_id": action_id,
                    "reward": reward,
                    "mask": mask
                })
            
            # 游戏状态更新
            game.current_player = (current_player + 1) % 4
            
        # 计算团队奖励
        team_reward = self.reward_shaper.compute_team_reward(
            game, pid, game.upgrade_amount
        )
        
        # 将团队奖励分配到每一步
        if memory:
            for i, transition in enumerate(memory):
                decay = max(0.2, 1.0 - i / len(memory))
                transition["reward"] += team_reward * decay
                
        return {"memory": memory, "game_result": game.ranking}
    
    def _sample_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        """采样动作"""
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.config.device)
        
        # 获取动作概率分布
        with torch.no_grad():
            probs = self.actor(state_tensor, mask_tensor)
        
        # 采样动作
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item()

    def _execute_action(self, game: GuandanGame, action_id: int) -> int:
        """执行动作并返回出牌数量"""
        player = game.players[game.current_player]
        action_struct = self.config.M_id_dict[action_id]  # 修改为使用config中的变量
        
        # 枚举合法出牌组合
        combos = enumerate_colorful_actions(action_struct, player.hand, game.active_level)
        cards_played = 0
        
        if combos:
            chosen_move = random.choice(combos)
            if chosen_move:
                # 执行出牌
                for card in chosen_move:
                    player.played_cards.append(card)
                    player.hand.remove(card)
                game.last_play = chosen_move
                game.last_player = game.current_player
                cards_played = len(chosen_move)
                game.recent_actions[game.current_player] = list(chosen_move)
            else:
                # Pass
                game.pass_count += 1
                game.recent_actions[game.current_player] = ['Pass']
        else:
            # Pass
            game.pass_count += 1
            game.recent_actions[game.current_player] = ['Pass']
            
        return cards_played

# === 6. 主训练循环 ===
def main(
    config: Config,
    actor: ImprovedActor,
    critic: ImprovedCritic,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    start_epoch: int,
    save_dir: Path,
    log_file: Path
):    
    # 1. 加载配置
    print(f"Action dimension: {config.action_dim}")
    
    # 4. 异步运行器（增加reward_shaper参数）
    reward_shaper = RewardShaper(config)
    runner = AsyncRunner(config, actor, reward_shaper)
    
    # 5. 训练循环
    entropy_coef = config.entropy_coef
            
    def log_info(msg: str):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(f"{msg}\n")
                    
    log_info(f"开始训练 - Action dimension: {config.action_dim}")
            
    try:
        for epoch in range(start_epoch, config.num_epochs):
            # 衰减熵系数
            entropy_coef = max(
                config.min_entropy_coef,
                entropy_coef * config.entropy_decay
            )
            
            # 收集数据
            all_memories = []
            with mp.Pool(config.num_envs) as pool:
                results = pool.map(runner.run_episode, range(4))
                for result in results:
                    all_memories.extend(result["memory"])
                    
            # 更新策略
            for _ in range(config.update_steps):
                # 采样批次
                batch_indices = np.random.choice(
                    len(all_memories),
                    config.batch_size,
                    replace=False
                )
                batch = [all_memories[i] for i in batch_indices]
                
                # PPO更新
                policy_loss, value_loss = update_policy(
                    actor, critic,
                    actor_optimizer, critic_optimizer,
                    batch, config
                )
            
            # 在训练循环中使用
            if (epoch + 1) % 50 == 0:
                log_info(
                    f"Epoch {epoch + 1}: "
                    f"PolicyLoss={policy_loss:.4f}, "
                    f"ValueLoss={value_loss:.4f}, "
                    f"EntropyCoef={entropy_coef:.4f}"
                )
                
            # 保存模型
            if (epoch + 1) % 200 == 0:
                save_models(actor=actor,critic=critic,actor_optimizer=actor_optimizer,critic_optimizer=critic_optimizer,epoch=epoch + 1)
                
    except KeyboardInterrupt:
        log_info("训练被手动中断")
        # 保存最后的模型
        save_models(actor=actor,critic=critic,actor_optimizer=actor_optimizer,critic_optimizer=critic_optimizer,epoch=epoch + 1)
        
    except Exception as e:
        log_info(f"训练发生错误: {str(e)}")


def update_policy(
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    batch: List[Dict],
    config: Config
) -> tuple:
    """PPO策略更新"""
    # 准备数据
    states = torch.FloatTensor([s['state'] for s in batch]).to(config.device)
    actions = torch.LongTensor([s['action_id'] for s in batch]).to(config.device)
    rewards = torch.FloatTensor([s['reward'] for s in batch]).to(config.device)
    masks = torch.FloatTensor([s['mask'] for s in batch]).to(config.device)
    
    # 计算优势函数
    with torch.no_grad():
        values = critic(states).squeeze()
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        
        # GAE计算
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + config.gamma * next_values[t] - values[t]
            gae = delta + config.gamma * config.gae_lambda * gae
            advantages[t] = gae
            
        # 计算回报
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取旧策略的动作概率
        old_probs = actor(states, masks)
        old_log_probs = torch.log(old_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
    # PPO更新循环
    policy_loss = 0
    value_loss = 0
    entropy_loss = 0
    
    for _ in range(config.ppo_epochs):
        # 计算新策略
        probs = actor(states, masks)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 值函数损失
        value_pred = critic(states).squeeze()
        value_loss = F.smooth_l1_loss(value_pred, returns)
        
        # 熵正则化
        entropy_loss = -config.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # 更新网络
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        
        actor_optimizer.step()
        critic_optimizer.step()
        
    return policy_loss.item(), value_loss.item()

def save_models(
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    epoch: int
):
    """保存模型和训练状态"""
    os.makedirs('models', exist_ok=True)
    save_path = f'models/checkpoint_ep{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
    }, save_path)
    print(f"模型已保存: {save_path}")
    
def load_models(config: Config) -> tuple:
    """加载最新的模型和训练状态"""
    try:
        checkpoints = sorted(Path('models').glob('checkpoint_ep*.pth'))
        if not checkpoints:
            return None
            
        latest = str(checkpoints[-1])
        checkpoint = torch.load(latest, map_location=config.device)
        
        actor = ImprovedActor(config).to(config.device)
        critic = ImprovedCritic(config).to(config.device)
        
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        
        actor_optimizer = optim.AdamW(
            actor.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        critic_optimizer = optim.AdamW(
            critic.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        
        print(f"成功加载模型: {latest}")
        return actor, critic, actor_optimizer, critic_optimizer, start_epoch
        
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None

def plot_training_progress(
    log_file: Path,
    save_dir: Path
):
    """绘制训练进度图表"""
    import matplotlib.pyplot as plt
    
    epochs, policy_losses, value_losses, entropies = [], [], [], []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line:
                # 解析日志行
                parts = line.split()
                epochs.append(int(parts[1].rstrip(":")))
                policy_losses.append(float(parts[3].split("=")[1].rstrip(",")))
                value_losses.append(float(parts[4].split("=")[1].rstrip(",")))
                entropies.append(float(parts[5].split("=")[1]))
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(epochs, policy_losses, label='Policy Loss')
    plt.legend()
    plt.subplot(312)
    plt.plot(epochs, value_losses, label='Value Loss')
    plt.legend()
    plt.subplot(313)
    plt.plot(epochs, entropies, label='Entropy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_progress.png')
    plt.close()
        
def save_config(config: Config, save_dir: Path):
    """保存训练配置"""
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
        
if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f'runs/run_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化配置
    config = Config()
    save_config(config, save_dir)
    
    # 初始化日志
    log_file = save_dir / 'training.log'
    def log_info(msg: str):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    
    log_info(f"开始训练 - Action dimension: {config.action_dim}")
    
    try:
        # 尝试加载已有模型
        loaded_state = load_models(config)
        if loaded_state:
            actor, critic, actor_optimizer, critic_optimizer, start_epoch = loaded_state
            print(f"继续训练从 epoch {start_epoch}")
        else:
            actor = ImprovedActor(config).to(config.device)
            critic = ImprovedCritic(config).to(config.device)
            actor_optimizer = optim.AdamW(actor.parameters(), lr=config.lr)
            critic_optimizer = optim.AdamW(critic.parameters(), lr=config.lr)
            start_epoch = 0
            print("从头开始训练")
        
        # 创建 reward_shaper
        reward_shaper = RewardShaper(config)
        runner = AsyncRunner(config, actor, reward_shaper)
        
        # 开始训练
        # 开始训练（传递所有必要参数）
        main(
            config=config,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            start_epoch=start_epoch,
            save_dir=save_dir,
            log_file=log_file
        )
        
        # 训练结束，绘制进度图表
        plot_training_progress(log_file, save_dir)
        
    except Exception as e:
        log_info(f"训练异常终止: {str(e)}")
        raise

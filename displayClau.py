"""
display.py
掼蛋AI对局结果与模型评测辅助工具。支持批量评测模型胜率、排名、测试多模型等功能，并自带演示主流程。
依赖核心对局逻辑。
"""
import random
import os
import numpy as np
import torch.optim as optim
from pathlib import Path
from collections import Counter, defaultdict
from get_actions import enumerate_colorful_actions, CARD_RANKS, SUITS, encode_hand_108

try:
    from c_rule import Rules  # 导入 Cython 版本
except ImportError:
    from rule import Rules  # 退回 Python 版本
try:
    from c_give_cards import create_deck, shuffle_deck, deal_cards
except ImportError:
    from give_cards import create_deck, shuffle_deck, deal_cards
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

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


local_backbone = SharedBackbone().to(device)
local_actor = ImprovedResNetActor(local_backbone).to(device)
local_critic = ImprovedResNetCritic(local_backbone).to(device)

    # 优化器设置
    # 优化器设置 - 避免参数重复分组
optimizer_params = [
    {'params': local_backbone.parameters(), 'lr': 5e-5},
    {'params': [p for n, p in local_actor.named_parameters() if not n.startswith('backbone.')], 'lr': 3e-5},
    {'params': [p for n, p in local_critic.named_parameters() if not n.startswith('backbone.')], 'lr': 1e-4}
]

local_optimizer = optim.Adam(optimizer_params)

def load_checkpoint(backbone, actor, critic, optimizer, model_path="models"):
    model_files = Path(model_path)
    
    if model_files.exists():
        print(f"找到checkpoint文件:{model_path}")
    else:
        print(f"未找到checkpoint文件:{model_path}")
        return 0
    
    # 加载checkpoint到CPU，然后移动到GPU
    checkpoint = torch.load(model_files, map_location='cpu', weights_only=True)
    
    # 加载模型权重并移动到GPU
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.to(device)
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.to(device)
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 将优化器状态移动到GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    ep = checkpoint['episode']
    print(f"加载checkpoint文件: {model_files}")
    
    # 设置模型为评估模式
    backbone.eval()
    actor.eval()
    critic.eval()
    
    local_actor = actor
    local_critic = critic
    local_backbone = backbone

    return ep

class Player:
    def __init__(self, hand):
        """
        程序里的玩家是从0开始的，输出时会+1
        """
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌
        self.last_played_cards = []


class GuandanGame:
    def __init__(self, user_player=None, active_level=None, verbose=True, print_history=False, test=False, model_path="models/checkpoint_ep100.pth"):
        # **两队各自的级牌**
        self.print_history = print_history
        self.active_level = active_level if active_level else random.choice(range(2, 15)) # 随机初始级牌(2-A)
                
        # 历史记录，记录最近 300 轮的出牌情况（每轮包含 4 个玩家的出牌）
        self.history = []
        # **只传当前局的有效级牌**
        self.rules = Rules(self.active_level)  # 初始化规则引擎
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]  # 发牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = user_player - 1 if user_player else None  # 转换为索引（0~3）
        self.ranking = []  # 存储出完牌的顺序
        self.recent_actions = [[], [], [], []]
        self.verbose = verbose  # 控制是否输出文本
        self.team_1 = {0, 2}
        self.team_2 = {1, 3}
        self.is_free_turn = True
        self.jiefeng = False
        self.winning_team = 0
        self.is_game_over = False
        self.upgrade_amount = 0
        self.model_path=model_path
        
        # 设置测试模式
        self.test = test
        
        self.backbone = local_backbone
        self.actor = local_actor
        self.critic = local_critic
        
        # 优化器分组设置
        self.optimizer = local_optimizer
            
        # 使用传入的模型或加载新模型
        #self.initial_ep = load_checkpoint(self.backbone, self.actor, self.critic, self.optimizer, self.model_path)

        # **手牌排序**
        for player in self.players:
            player.hand = self.sort_cards(player.hand)

    def log(self, message):
        """控制是否打印消息"""
        if self.verbose:
            print(message)

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def play_turn(self):
        """执行当前玩家的回合"""

        player = self.players[self.current_player]  # 获取当前玩家对象

        # **计算当前仍有手牌的玩家数**
        active_players = 4 - len(self.ranking)

        # **如果 Pass 的人 == "当前有手牌的玩家数 - 1"，就重置轮次**
        if self.pass_count >= (active_players - 1) and self.current_player not in self.ranking:
            if self.jiefeng:
                first_player = self.ranking[-1]
                teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1
                self.log(f"\n🆕 轮次重置！玩家 {teammate + 1} 接风。\n")
                self.recent_actions[self.current_player] = []  # 记录空列表
                self.current_player = (self.current_player + 1) % 4
                self.last_play = None  # ✅ 允许新的自由出牌
                self.pass_count = 0  # ✅ Pass 计数归零
                self.is_free_turn = True
                self.jiefeng = False
            else:
                self.log(f"\n🆕 轮次重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
                self.last_play = None  # ✅ 允许新的自由出牌
                self.pass_count = 0  # ✅ Pass 计数归零
                self.is_free_turn = True

        if self.user_player == self.current_player:
            result = self.user_play(player)
        else:
            if self.test and (self.current_player == 0 or self.current_player == 2):  #team_1 0,2  team_2 1,3 
                result = self.actor_play(player)
            else:
                result = self.ai_play(player)

        # 记录历史（每轮结束时）
        if self.current_player == 0 and any(action != ['None'] for action in self.recent_actions):
            round_history = [self.recent_actions[i].copy() for i in range(4)]
            self.history.append(round_history)
                
            # 重置最近动作（使用深拷贝避免引用问题）
            self.recent_actions = [['None'] for _ in range(4)]
            
            if len(self.history) > 20:
                self.history.pop(0)
            
        return result


    def get_possible_moves(self, player_hand):
        """获取所有可能的合法出牌，包括各种牌型：单张、对子、顺子、连对、钢板、三带二等，并支持级牌红桃作为赖子"""
        
        possible_moves = []
        # 确保player_hand是列表类型
        if not isinstance(player_hand, list):
            player_hand = []
            
        # 获取所有非赖子牌的点数（排除红桃级牌）
        non_wildcards = [card for card in player_hand if not self.rules._is_wildcard(card)]
        hand_points = [self.rules.get_rank(card) for card in non_wildcards]  # 仅点数（去掉花色）
        hand_counter = Counter(hand_points)  # 统计点数出现次数
        unique_points = sorted(set(hand_points))  # 仅保留唯一点数，排序
        
        # 查找红桃级牌（赖子）
        wildcard_point = self.rules.level_card  # 直接使用规则中的级牌点数
        wildcard_cards = [card for card in player_hand if self.rules._is_wildcard(card)]
        wildcard_count = len(wildcard_cards)
        
        # 确保wildcard_cards不为None
        if wildcard_cards is None:
            wildcard_cards = []
            wildcard_count = 0
            
        # 1. **基础牌型：单张、对子、三条等**
        for size in [1, 2, 3]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                current_valid, current_type = self.rules.is_valid_play(move)
                if current_valid:
                    # 只有当last_play为空或当前牌型能压过last_play时才添加
                    if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                        possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])

        # 2. **炸弹牌型（4张及以上同点数）**
        # 遍历每个点数，生成可能的炸弹组合
        for point, count in hand_counter.items():
            # 计算该点数可用的赖子数量
            available_wildcards = wildcard_count
            
            # 生成4张及以上的炸弹
            for bomb_size in range(4, count + available_wildcards + 1):
                # 获取该点数的所有非赖子牌
                non_wild = [card for card in player_hand if self.rules.get_rank(card) == point]
                # 需要补足的赖子数量
                need_wildcards = bomb_size - len(non_wild)
                
                # 如果赖子足够且数量合理
                if need_wildcards >= 0 and need_wildcards <= available_wildcards:
                    # 使用赖子补足
                    wild_used = wildcard_cards[:need_wildcards]
                    move = non_wild + wild_used
                    current_valid, current_type = self.rules.is_valid_play(move)
                    if current_valid:
                        # 只有当last_play为空或当前牌型能压过last_play时才添加
                        if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                            possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])
        
        # 3. **纯赖子炸弹（4张及以上）**can_beat
        if wildcard_count >= 4:
            for size in range(4, wildcard_count + 1):
                move = wildcard_cards[:size]
                current_valid, current_type = self.rules.is_valid_play(move)
                if current_valid:
                    # 只有当last_play为空或当前牌型能压过last_play时才添加
                    if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                        possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])

        # 4. **顺子（5张及以上）**
        min_straight_len = 5
        max_straight_len = len(unique_points) + wildcard_count
        for length in range(min_straight_len, max_straight_len + 1):
            # 生成所有可能的连续点数序列
            for start in range(0, len(unique_points) - length + wildcard_count + 1):
                # 获取连续点数序列
                seq = unique_points[start:start + length]
                
                # 检查序列是否连续（允许用赖子填补空缺）
                gap_count = 0
                for j in range(1, len(seq)):
                    gap = seq[j] - seq[j-1] - 1
                    if gap > 0:
                        gap_count += gap
                
                # 如果空缺数不超过赖子数量，则可以组成顺子
                if gap_count <= wildcard_count and gap_count >= 0:
                    # 生成可能的顺子组合（考虑赖子）
                    move = self._generate_straight_with_wildcards(
                        player_hand, seq, wildcard_cards, wildcard_point
                    )
                    current_valid, current_type = self.rules.is_valid_play(move)
                    if current_valid:
                        # 只有当last_play为空或当前牌型能压过last_play时才添加
                        if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                            possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])

        # 5. **连对 (3对以上连续)**
        min_pair_chain_len = 3
        max_pair_chain_len = len(unique_points) // 2
        for length in range(min_pair_chain_len, max_pair_chain_len + 1):
            for i in range(len(unique_points) - length + 1):
                seq = unique_points[i:i + length]
                
                # 检查每张牌是否至少有两张（允许用赖子补足）
                missing_pairs = 0
                for p in seq:
                    if hand_counter[p] < 2:
                        missing_pairs += (2 - hand_counter[p])
                
                # 如果缺失的对子数不超过赖子数量，则可以组成连对
                if missing_pairs <= wildcard_count:
                    move = self._generate_pair_chain_with_wildcards(
                        player_hand, seq, wildcard_cards, wildcard_point
                    )
                    current_valid, current_type = self.rules.is_valid_play(move)
                    if current_valid:
                        # 只有当last_play为空或当前牌型能压过last_play时才添加
                        if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                            possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])

        # 6. **钢板 (2组以上连续三张)**
        min_trio_chain_len = 2
        max_trio_chain_len = len(unique_points) // 3
        for length in range(min_trio_chain_len, max_trio_chain_len + 1):
            for i in range(len(unique_points) - length + 1):
                seq = unique_points[i:i + length]
                
                # 检查每张牌是否至少有三张（允许用赖子补足）
                missing_trios = 0
                for p in seq:
                    if hand_counter[p] < 3:
                        missing_trios += (3 - hand_counter[p])
                
                # 如果缺失的三张数不超过赖子数量，则可以组成钢板
                if missing_trios <= wildcard_count:
                    move = self._generate_trio_chain_with_wildcards(
                        player_hand, seq, wildcard_cards, wildcard_point
                    )
                    current_valid, current_type = self.rules.is_valid_play(move)
                    if current_valid:
                        # 只有当last_play为空或当前牌型能压过last_play时才添加
                        if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                            possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])

        # 7. **三带二**
        # 找出所有可能的三张组合（允许用赖子补足）
        trios = []
        for p, count in hand_counter.items():
            if count + min(wildcard_count, 3 - count) >= 3:
                trios.append(p)
        
        # 找出所有可能的对子组合（允许用赖子补足）
        pairs = []
        for p, count in hand_counter.items():
            if count + min(wildcard_count, 2 - count) >= 2:
                pairs.append(p)
        
        # 使用副本避免修改原始列表
        hand_copy = player_hand[:]
        wildcards_copy = wildcard_cards[:]
        
        for trio_point in trios:
            # 获取三张牌（可能包含赖子）
            trio_cards = self._get_cards_with_wildcards(
                hand_copy, trio_point, 3, wildcards_copy, wildcard_point
            )
            if not trio_cards:
                continue
                
            # 记录三张部分使用的赖子
            trio_wildcards_used = [card for card in trio_cards if self.rules._is_wildcard(card)]
            trio_non_wildcards = [card for card in trio_cards if not self.rules._is_wildcard(card)]
            
            # 从副本中移除已使用的牌
            for card in trio_non_wildcards:
                if card in hand_copy:
                    hand_copy.remove(card)
            for card in trio_wildcards_used:
                if card in wildcards_copy:
                    wildcards_copy.remove(card)
            
            for pair_point in pairs:
                # 跳过与三张牌相同的点数
                if pair_point == trio_point:
                    continue
                    
                # 获取对子牌（可能包含赖子）
                pair_cards = self._get_cards_with_wildcards(
                    hand_copy, pair_point, 2, wildcards_copy, wildcard_point
                )
                if not pair_cards:
                    continue
                    
                # 记录对子部分使用的赖子
                pair_wildcards_used = [card for card in pair_cards if self.rules._is_wildcard(card)]
                pair_non_wildcards = [card for card in pair_cards if not self.rules._is_wildcard(card)]
                
                # 检查赖子是否重复使用
                wildcards_conflict = any(card in trio_wildcards_used for card in pair_wildcards_used)
                if wildcards_conflict:
                    # 恢复对子部分使用的牌
                    for card in pair_non_wildcards:
                        if card not in hand_copy:
                            hand_copy.append(card)
                    continue
                    
                # 从副本中移除对子牌
                for card in pair_non_wildcards:
                    if card in hand_copy:
                        hand_copy.remove(card)
                for card in pair_wildcards_used:
                    if card in wildcards_copy:
                        wildcards_copy.remove(card)
                
                move = trio_cards + pair_cards
                current_valid, current_type = self.rules.is_valid_play(move)
                if current_valid:
                    # 只有当last_play为空或当前牌型能压过last_play时才添加
                    if not self.last_play or self.rules.can_beat(self.last_play, move)[0]:
                        possible_moves.append([move, self.rules.can_beat(self.last_play, move)[1], self.rules.can_beat(self.last_play, move)[2]])
                
                # 恢复对子部分使用的牌
                for card in pair_non_wildcards:
                    if card not in hand_copy:
                        hand_copy.append(card)
                for card in pair_wildcards_used:
                    if card not in wildcards_copy:
                        wildcards_copy.append(card)
            
            # 恢复三张部分使用的牌
            for card in trio_non_wildcards:
                if card not in hand_copy:
                    hand_copy.append(card)
            for card in trio_wildcards_used:
                if card not in wildcards_copy:
                    wildcards_copy.append(card)

        return possible_moves

    def _get_cards_by_point(self, hand, point, count):
        """从手牌中获取指定点数的牌"""
        cards = []
        hand_copy = hand[:]
        for card in hand_copy:
            if self.rules.get_rank(card) == point and len(cards) < count:
                cards.append(card)
                hand_copy.remove(card)
        return cards

    def _generate_straight_with_wildcards(self, hand, seq, wildcards, wildcard_point):
        """生成包含赖子的顺子"""
        move = []
        hand_copy = hand[:]
        wildcards_copy = wildcards[:]
        
        # 尝试填充序列中的每个点数
        for p in seq:
            found = False
            # 先尝试使用真实牌
            for card in hand_copy:
                if self.rules.get_rank(card) == p:
                    move.append(card)
                    hand_copy.remove(card)
                    found = True
                    break
            
            # 如果没有找到真实牌，使用赖子
            if not found and wildcards_copy:
                move.append(wildcards_copy.pop(0))
        
        # 检查生成的顺子是否合法（点数不重复）
        points = [self.rules.get_rank(card) for card in move]
        if len(points) != len(set(points)):
            return None  # 点数重复，非法顺子
        
        return move if len(move) == len(seq) else None

    def _generate_pair_chain_with_wildcards(self, hand, seq, wildcards, wildcard_point):
        """生成包含赖子的连对"""
        move = []
        hand_copy = hand[:]
        wildcards_copy = wildcards[:]
        
        # 尝试为序列中的每个点数生成对子
        for p in seq:
            # 收集该点数的牌
            cards = []
            for card in hand_copy[:]:
                if self.rules.get_rank(card) == p:
                    cards.append(card)
                    hand_copy.remove(card)
                    if len(cards) == 2:
                        break
            
            # 补足对子
            while len(cards) < 2 and wildcards_copy:
                cards.append(wildcards_copy.pop(0))
            
            if len(cards) < 2:
                return None  # 无法补足对子
            
            move.extend(cards)
        
        return move

    def _generate_trio_chain_with_wildcards(self, hand, seq, wildcards, wildcard_point):
        """生成包含赖子的钢板"""
        move = []
        hand_copy = hand[:]
        wildcards_copy = wildcards[:]
        
        # 尝试为序列中的每个点数生成三张
        for p in seq:
            # 收集该点数的牌
            cards = []
            for card in hand_copy[:]:
                if self.rules.get_rank(card) == p:
                    cards.append(card)
                    hand_copy.remove(card)
                    if len(cards) == 3:
                        break
            
            # 补足三张
            while len(cards) < 3 and wildcards_copy:
                cards.append(wildcards_copy.pop(0))
            
            if len(cards) < 3:
                return None  # 无法补足三张
            
            move.extend(cards)
        
        return move

    def _get_cards_with_wildcards(self, hand, point, count, wildcards, wildcard_point):
        """获取指定点数的牌，可使用赖子补足"""
        cards = []
        hand_copy = hand[:]
        wildcards_copy = wildcards[:]
        
        # 收集真实牌
        for card in hand_copy[:]:
            if self.rules.get_rank(card) == point:
                cards.append(card)
                hand_copy.remove(card)
                if len(cards) == count:
                    break
        
        # 用赖子补足
        while len(cards) < count and wildcards_copy:
            cards.append(wildcards_copy.pop(0))
        
        return cards if len(cards) == count else None
    
    def ai_play(self, player):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        # **如果玩家已经打完，仍然记录一个空列表，然后跳过**
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            # 移动到下一个未出完牌的玩家
            next_player = (self.current_player + 1) % 4
            while next_player in self.ranking and not self.is_game_over:
                next_player = (next_player + 1) % 4
            self.current_player = next_player
            return self.check_game_over()

        player_hand = player.hand

        possible_moves = self.get_possible_moves(player_hand)
        if not self.is_free_turn:
            possible_moves.append([])

        if not possible_moves:
            self.log(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
        else:
            chosen_move = random.choice(possible_moves) # 随机选择一个合法的牌型
            
            if chosen_move:
                chosen_move = chosen_move[0]
                
            if not chosen_move:
                self.log(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
            else:
                # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                self.last_play = chosen_move
                self.last_player = self.current_player
                self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
                
                # 确保只移除实际存在于手牌中的牌
                valid_move = []
                for card in chosen_move:
                    if card in player_hand:
                        valid_move.append(card)
                        player_hand.remove(card)
                    else:
                        # 如果是赖子牌，使用其点数创建新卡牌
                        if self.rules._is_wildcard(card):
                            wildcard_rank = self.rules.level_card
                            wildcard_suit = "红桃"
                            new_card = f"{wildcard_suit}{RANKS[wildcard_rank-2]}"
                            if new_card in player_hand:
                                valid_move.append(new_card)
                                player_hand.remove(new_card)
                            else:
                                # 如果找不到对应的级牌，尝试使用其他花色
                                for suit in SUITS:
                                    new_card = f"{suit}{RANKS[wildcard_rank-2]}"
                                    if new_card in player_hand:
                                        valid_move.append(new_card)
                                        player_hand.remove(new_card)
                                        break
                
                # 更新出牌记录
                if valid_move:
                    self.last_play = valid_move
                    self.last_player = self.current_player
                    #self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(valid_move)}")
                    player.played_cards.extend(valid_move)
                    self.recent_actions[self.current_player] = valid_move
                    self.jiefeng = False
                
                self.recent_actions[self.current_player] = list(chosen_move)  # 记录出牌
                self.jiefeng = False
                if not player_hand:  # 玩家出完牌
                    self.log(f"🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                    self.ranking.append(self.current_player)
                    #if len(self.ranking)<=2:
                    #    self.jiefeng=True

                self.pass_count = 0
                if not player_hand:
                    self.pass_count -= 1

                if self.is_free_turn:
                    self.is_free_turn = False

        player.last_played_cards = self.recent_actions[self.current_player]
        
        game_over = self.check_game_over()
        
        if not game_over:
            next_player = (self.current_player + 1) % 4
            self.current_player = next_player
        
            while next_player in self.ranking:
                next_player = (next_player + 1) % 4
                self.current_player = next_player
                                    
        return game_over

    def actor_play(self, player):
        combos = []
        chosen_move_temp = 0
        player = self.players[self.current_player] 
        action_id = 0

        # 0. 直接计算状态和掩码（移除缓存机制）
        state = self._get_obs()
    
        # 创建状态张量并移到GPU
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 创建合法动作掩码
        mask_tensor = torch.zeros(action_dim, dtype=torch.float32, device=device)
        
        # 1. 获取合法动作列表
        combos = self.get_possible_moves(player.hand)
        
        # 创建临时的action_id_val与combos对照表
        action_id_to_combo_map = {}
        if combos:
            # 循环计算每个combo对应的action_id
            for combo in combos:
                action_id_val= self.rules.cards_to_mod_id(combo[0])
                #action_id_val = self.rules.PLAY_MAPPING.get(combo[1], -1)
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
        probs, logits = self.actor(state_tensor, mask_tensor)
        action_id = torch.multinomial(probs, 1).item()

        if action_id in action_id_to_combo_map:
            chosen_move = action_id_to_combo_map[action_id]
                                
            if not chosen_move:
                self.log(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
            else:
                # 如果 chosen_move 不为空，继续进行正常的出牌逻辑
                self.last_play = chosen_move
                self.last_player = self.current_player
                for card in chosen_move:
                    player.played_cards.append(card)
                    player.hand.remove(card)
                self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
                self.recent_actions[self.current_player] = list(chosen_move)  # 记录出牌
                self.jiefeng = False
                if not player.hand:  # 玩家出完牌
                    self.log(f"🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                    self.ranking.append(self.current_player)
                    if len(self.ranking) <= 2:
                        self.jiefeng = True

                self.pass_count = 0
                if not player.hand:
                    self.pass_count -= 1

                if self.is_free_turn:
                    self.is_free_turn = False
        else:
            self.log(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']  # 记录 Pass
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()


    def user_play(self, player):
        """用户出牌逻辑"""
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            self.current_player = (self.current_player + 1) % 4
            return self.check_game_over()

        self.get_ai_suggestions(player)
        while True:
            self.show_user_hand()  # 显示手牌
            choice = input("\n请选择要出的牌（用空格分隔），或直接回车跳过（PASS）： ").strip()

            # **用户选择 PASS**
            if choice == "" or choice.lower() == "pass":
                if self.is_free_turn:
                    print("❌ 你的输入无效，自由回合必须出牌！")
                    continue
                print(f"玩家 {self.current_player + 1} 选择 PASS")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']  # ✅ 记录 PASS
                break

            # **解析用户输入的牌**
            selected_cards = choice.split()

            # **检查牌是否在手牌中**
            if not all(card in player.hand for card in selected_cards):
                print("❌ 你的输入无效，请确保牌在你的手牌中！")
                continue  # 重新输入

            # **检查牌是否合法**
            if not self.rules.is_valid_play(selected_cards):
                print("❌ 你的出牌不符合规则，请重新选择！")
                continue  # 重新输入

            last_action = self.map_cards_to_action(self.last_play, M, self.active_level)
            chosen = self.map_cards_to_action(selected_cards, M, self.active_level)
            # **检查是否能压过上一手牌**
            if not self.can_beat(chosen, last_action):
                print("❌ 你的牌无法压过上一手牌，请重新选择！")
                continue  # 重新输入

            # **成功出牌**
            for card in selected_cards:
                player.played_cards.append(card)
                player.hand.remove(card)  # 从手牌中移除
            self.last_play = selected_cards  # 记录这次出牌
            self.last_player = self.current_player  # 记录是谁出的
            self.recent_actions[self.current_player] = list(selected_cards)  # 记录出牌历史
            self.jiefeng = False
            print(f"玩家 {self.current_player + 1} 出牌: {' '.join(selected_cards)}")

            # **如果手牌为空，玩家出完所有牌**
            if not player.hand:
                print(f"🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.ranking.append(self.current_player)
                if len(self.ranking) <= 2:
                    self.jiefeng = True

            # **出牌成功，Pass 计数归零**
            self.pass_count = 0
            if not player.hand:
                self.pass_count -= 1
            if self.is_free_turn:
                self.is_free_turn = False
            break

        # **切换到下一个玩家**
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4

        return self.check_game_over()

    def get_ai_suggestions(self, player):
        # --- Get AI Suggestions ---
        combos = []
        chosen_move_temp = 0
        player = self.players[0]
        action_id = 0

        # 0. 直接计算状态和掩码（移除缓存机制）
        state = self._get_obs()
    
        # 创建状态张量并移到GPU
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 创建合法动作掩码
        mask_tensor = torch.zeros(action_dim, dtype=torch.float32, device=device)
        
        # 1. 获取合法动作列表
        combos = self.get_possible_moves(player.hand)
        
        # 创建临时的action_id_val与combos对照表
        action_id_to_combo_map = {}
        if combos:
            # 循环计算每个combo对应的action_id
            for combo in combos:
                action_id_val = self.rules.PLAY_MAPPING.get(combo[1], -1)
                if action_id_val != -1:
                    action_id_val = action_id_val*10000 + combo[2]
                    action_id_val = action_id_val % 1023
                
                # 添加到对照表
                if 0 <= action_id_val < action_dim:
                    # 使用combo[0]作为键值（出牌组合）
                    action_id_to_combo_map[action_id_val] = combo[0]
                    mask_tensor[action_id_val] = 1.0
        else:
            # 没有合法动作时只允许Pass
            mask_tensor[0] = 1.0
            action_id_to_combo_map[0] = []  # Pass动作

        # Ensure the actor model is accessible, e.g., self.actor if it's part of the class
        # Or just 'actor' if it's loaded globally as in your original file example
        #global actor  # Assuming actor is loaded globally
        with torch.no_grad():  # Disable gradient calculation for inference
            # Note: The actor output 'probs' is already after a softmax over ALL actions
            all_probs, _ = self.actor(state_tensor, mask_tensor)  # 解包元组，只取概率部分

        # Get top 3 suggestions (indices and their original probabilities)
        # We use the original probabilities from the full softmax to find the top K
        top_k_orig_probs, top_k_indices = torch.topk(all_probs, k=3, dim=-1)

        # --- Apply Softmax to ONLY the top K probabilities for relative comparison ---
        # Detach is good practice here although not strictly necessary with no_grad()
        # We only apply softmax if there are positive probabilities to normalize
        valid_top_k_probs = top_k_orig_probs[top_k_orig_probs > 0]  # Filter out zero probabilities if any
        if valid_top_k_probs.numel() > 0:
            # Apply softmax to the non-zero probabilities of the top k actions
            normalized_top_k_probs_tensor = F.softmax(valid_top_k_probs, dim=-1)
            # Create a placeholder for normalized probabilities matching top_k_indices size
            normalized_top_k_probs = torch.zeros_like(top_k_orig_probs)
            # Fill in the normalized probabilities where the original probs were positive
            normalized_top_k_probs[top_k_orig_probs > 0] = normalized_top_k_probs_tensor
        else:
            # Handle case where all top k probabilities are zero (e.g., mask filtered everything)
            normalized_top_k_probs = torch.zeros_like(top_k_orig_probs)

        print("\n--- AI 建议 ---")
        for i in range(top_k_indices.size(1)):
            action_id = top_k_indices[0, i].item()
            # Use the normalized probability for display
            normalized_prob = normalized_top_k_probs[0, i].item()

            # We still check original probability > 0 to decide if it was a valid move initially
            if top_k_orig_probs[0, i].item() > 0:
                points_str = action_id_to_combo_map[action_id]
                # Display the normalized probability
                print(f"建议 {i + 1}: {points_str} - 相对概率: {normalized_prob:.2%}")
            else:
                # If original probability was 0, it wasn't a valid move
                print(f"建议 {i + 1}: (无有效动作)")

        print("-----------------------------")

    def check_game_over(self):
        """检查游戏是否结束"""
        # **如果有 2 个人出完牌，并且他们是同一队伍，游戏立即结束**
        if len(self.ranking) >= 2:
            # 检查前两名是否同队
            if (self.ranking[0] in self.team_1 and self.ranking[1] in self.team_1) or (
                    self.ranking[0] in self.team_2 and self.ranking[1] in self.team_2):
                # 补全剩余玩家排名
                remaining_players = [i for i in range(4) if i not in self.ranking]
                self.ranking.extend(remaining_players)
                self.update_level()
                self.is_game_over = True
                return True

        # **如果 3 人出完了，自动补全最后一名，游戏结束**
        if len(self.ranking) == 3:
            self.ranking.append(next(i for i in range(4) if i not in self.ranking))  # 找出最后一个玩家
            self.update_level()
            self.is_game_over = True
            return True

        return False

    def update_level(self):
        """升级级牌并确保在2到A之间循环"""
        first_player = self.ranking[0]  # 第一个打完牌的玩家
        winning_team = 1 if first_player in self.team_1 else 2
        self.winning_team = winning_team
        
        # 确定队友
        teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1

        # 找到队友在排名中的位置
        teammate_position = self.ranking.index(teammate)

        # 头游 + 队友的名次，确定得分
        upgrade_map = {1: 3, 2: 2, 3: 1}  # 头游 + (队友的名次) 对应的升级规则
        upgrade_amount = upgrade_map[teammate_position]
        self.upgrade_amount = upgrade_amount

        # 计算新的级牌（2-14对应2到A）
        new_level = self.active_level + upgrade_amount
        
        # 确保级牌在2到A之间循环
        if new_level > 14:  # 超过A级
            new_level = new_level % 13 + 2  # 循环回到2级
            if new_level < 2:
                new_level = 2
        
        self.active_level = new_level

        self.log(f"🏆 {winning_team} 号队伍获胜！得 {upgrade_amount} 分 \n")
        self.log(f"新的级牌为：{RANKS[self.active_level-2]}")
        # 显示最终排名
        ranks = ["头游", "二游", "三游", "末游"]
        for i, player in enumerate(self.ranking):
            self.log(f"{ranks[i]}：玩家 {player + 1}")
        self.log("局终\n")

    def reset_game(self):
        """重置游戏状态，重新洗牌和发牌"""
        # 重置游戏状态
        self.history = []
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]
        self.current_player = 0
        self.last_play = None
        self.last_player = -1
        self.pass_count = 0
        self.ranking = []
        self.recent_actions = [[], [], [], []]
        self.is_free_turn = True
        self.jiefeng = False
        self.is_game_over = False
        
        # 重新排序手牌
        for player in self.players:
            player.hand = self.sort_cards(player.hand)

    def play_game(self):
        """执行一整局游戏"""
        self.log(f"\n🎮 游戏开始！当前级牌：{self.active_level}")

        while True:
            if self.play_turn():
                if self.current_player != 0:
                    round_history = [self.recent_actions[i] for i in range(4)]
                    self.history.append(round_history)
                if self.print_history:
                    for i in range(len(self.history)):
                        self.log(self.history[i])
                # 游戏结束后重置牌局
                #self.reset_game()
                break

    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.players[self.user_player].hand
        print("\n你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"场上最新出牌：{' '.join(self.last_play)}\n")

    def _get_obs(self):
        """
        构造状态向量，总共 3050 维
        """
        obs = np.zeros(1430)
        
        # 1️⃣ 当前玩家手牌 (108)
        obs[:108]=encode_hand_108(self.players[self.current_player].hand)
        offset = 108

        # 2️⃣ 其他玩家手牌数量 (3)
        for i, player in enumerate(self.players):
            if i != self.current_player:
                obs[offset + i] = min(len(player.hand), 26) / 26.0
        offset += 3

        # 3️⃣ 最近动作 (108 * 4 = 432)
        for i, player in enumerate(self.players):
            obs[offset + i * 108 : offset + (i + 1) * 108] = encode_hand_108(player.last_played_cards)
        offset += 108 * 4

        # 4️⃣ 其他玩家已出牌 (108 * 3 = 324)
        for i, player in enumerate(self.players):
            if i != self.current_player:
                obs[offset + i * 108 : offset + (i + 1) * 108] = encode_hand_108(player.played_cards)
        offset += 108 * 3

        # 5️⃣ 当前级牌 (13)
        obs[offset + self.level_card_to_index(self.active_level)] = 1
        offset += 13
        
        # 6️⃣ 级牌数值 (1) - 新增
        obs[offset] = self.active_level / 14.0  # 归一化到0-1范围
        offset += 1

        # 7️⃣ 最近 20 步动作历史 (108 * 5 = 2160)2160
        HISTORY_LEN = 5
        history_flat = []

        # 展平所有轮次中的动作
        for round in self.history:
            for action in round:
                history_flat.append(action)

        # 若不满 20，则在最前补空动作（表示“没人出牌”）
        while len(history_flat) < HISTORY_LEN:
            history_flat.insert(0, [])  # 用空动作填充

        # 取最后 20 个动作
        history_flat = history_flat[-HISTORY_LEN:]

        # 编码入 obs
        for i, action in enumerate(history_flat):
            start = offset + i * 108
            obs[start:start + 108] = encode_hand_108(action)
        offset += 108 * HISTORY_LEN

        # 8️⃣ 状态向量 (9)
        obs[offset:offset + 3] = self.compute_coop_status()
        obs[offset + 3:offset + 6] = self.compute_dwarf_status()
        obs[offset + 6:offset + 9] = self.compute_assist_status()
        offset += 9

        assert offset == 1430, f"⚠️ offset 计算错误: 预期 3050, 实际 {offset}"
        return obs

    def level_card_to_index(self, level_card):
        """
        级牌转换为 one-hot 索引 (2 -> 0, 3 -> 1, ..., A -> 12)
        """
        levels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return levels.index(str(level_card)) if str(level_card) in levels else 0

    def compute_coop_status(self):
        """
        计算协作状态：
        [1, 0, 0] -> 不能协作
        [0, 1, 0] -> 选择协作
        [0, 0, 1] -> 拒绝协作
        """
        return [1, 0, 0]  # 目前默认"不能协作"，后续可修改逻辑

    def compute_dwarf_status(self):
        """
        计算压制状态：
        [1, 0, 0] -> 不能压制
        [0, 1, 0] -> 选择压制
        [0, 0, 1] -> 拒绝压制
        """
        return [1, 0, 0]  # 目前默认"不能压制"，后续可修改逻辑

    def compute_assist_status(self):
        """
        计算辅助状态：
        [1, 0, 0] -> 不能辅助
        [0, 1, 0] -> 选择辅助
        [0, 0, 1] -> 拒绝辅助
        """
        return [1, 0, 0]  # 目前默认"不能辅助"，后续可修改逻辑


def test_multiple_models():  # 批量测试不同训练阶段的模型   # 统计胜率、名次等指标
    # 测试的episode范围
    start_ep = 6100
    end_ep = 8400
    step = 200
    test_eps = range(start_ep, end_ep + step, step)

    # 存储结果的字典
    results = {
        'episodes': [],
        'win_rates': [],
        'first_hand_rates': [],
        'yi_rates': [],
        'er_rates': [],
        'san_rates': []
    }

    n = 200 # 每个模型测试的场次
            
    for model_ep in test_eps:

        win = 0
        first = 0
        yi = 0
        er = 0
        san = 0

        # 确保在测试开始时设置随机种子
        torch.manual_seed(420)
        random.seed(420)
        np.random.seed(420)
    
        print(f"测试模型 checkpoint_ep{model_ep}.pth 开始...") #team_1 0,2  team_2 1,3         
        #加载新模型
        load_checkpoint(local_backbone, local_actor, local_critic, local_optimizer, model_path=f"models/checkpoint_ep{model_ep}.pth")
        
        for _ in range(n):
            try:
                game = GuandanGame(user_player=None, active_level=None,verbose=True, print_history=True,test=True,model_path=f"models/checkpoint_ep{model_ep}.pth")
                game.play_game()

                if game.winning_team == 1:
                    win += 1
                    if game.upgrade_amount == 3:
                        yi += 1
                    elif game.upgrade_amount == 2:
                        er += 1
                    else:
                        san += 1

                if game.ranking[0] == 0:
                    first += 1
                    
                game.reset_game()
                
            except Exception as e:
                print(f"测试模型 checkpoint_ep{model_ep}.pth 时出错: {str(e)}")
                continue

        # 存储结果
        results['episodes'].append(model_ep)
        results['win_rates'].append(win / n * 100)
        results['first_hand_rates'].append(first / n * 100)
        results['yi_rates'].append(yi / win * 100 if win > 0 else 0)
        results['er_rates'].append(er / win * 100 if win > 0 else 0)
        results['san_rates'].append(san / win * 100 if win > 0 else 0)
        print(f"测试模型 checkpoint_ep{model_ep}.pth结束")

        # 打印所有模型的汇总结果
        print("\n\n" + "=" * 80)
        print("所有模型测试结果汇总（团队）:")
        print("=" * 80)
        print(
            f"{'Episode':<10}{'胜率(%)':<10}{'第一手出完率(%)':<18}{'一二名(%)':<15}{'一三名(%)':<15}{'一四名(%)':<15}")
        print("-" * 80)
        for i in range(len(results['episodes'])):
            ep = results['episodes'][i]
            print(f"{ep:<10}{results['win_rates'][i]:<10.2f}{results['first_hand_rates'][i]:<18.2f}"
                f"{results['yi_rates'][i]:<15.2f}{results['er_rates'][i]:<15.2f}{results['san_rates'][i]:<15.2f}")

        print(results)


if __name__ == "__main__":
    #model_ep = 2000
    #load_checkpoint(local_backbone, local_actor, local_critic, local_optimizer, model_path=f"models/checkpoint_ep{model_ep}.pth")
    #game = GuandanGame(user_player=1, active_level=None, verbose=True, print_history=True)
    #game.play_game()
    test_multiple_models()



'''
根据代码中的训练配置和掼蛋游戏的复杂性，训练轮次建议如下：

1. 基础训练阶段（0-5,000轮）

模型开始学习基本出牌规则和简单牌型
主要掌握单牌、对子、三张等基础牌型
预期效果：Pass率显著下降，能完成30%左右的合法出牌
2. 中级训练阶段（5,000-15,000轮）

开始学习组合牌型（三带二、连对等）
炸弹使用策略逐渐形成
预期效果：能处理70%常见牌型，简单配合策略出现
3. 高级训练阶段（15,000-30,000轮）

掌握所有复杂牌型（飞机带翅膀、逢人配等）
形成初步的战术配合和牌力评估能力
预期效果：能达到业余高手水平，胜率超过规则型AI
4. 精调阶段（30,000+轮）

优化策略细节和特殊情况处理
形成稳定的风格和高级战术
预期效果：接近职业选手水平，春天率显著提升
关键训练指标观察点：

5,000轮：Pass率应降至30%以下
10,000轮：炸弹使用合理率超过50%
20,000轮：复杂牌型识别率超过80%
30,000轮：团队配合动作占比超过40%
建议每5,000轮进行一次模型评估，当连续3次评估胜率提升小于2%时，可考虑停止训练。完整训练通常需要2-3周（使用单卡GPU）。
'''

import numpy as np
from get_actions import enumerate_colorful_actions, CARD_RANKS, SUITS,encode_hand_108
import random
from collections import Counter, defaultdict
import streamlit as st
import time
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
import torch.optim as optim
import torch.nn.functional as F
import json
with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)
# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
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
actor = ActorNet()
actor.load_state_dict(torch.load("models/actor_ep200.pth"))
actor.eval()
class Player:
    def __init__(self, hand):
        """
        程序里的玩家是从0开始的，输出时会+1
        """
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌
        self.last_played_cards = []


class GuandanGame:
    def __init__(self, user_player=None, active_level=None):
        self.active_level = active_level if active_level else random.choice(range(2, 15))
        self.history = []
        self.rules = Rules(self.active_level)
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]
        self.current_player = 0
        self.last_play = None
        self.last_player = -1
        self.pass_count = 0
        self.user_player = user_player - 1 if user_player else None
        self.ranking = []
        self.recent_actions = [[], [], [], []]
        self.team_1 = {0, 2}
        self.team_2 = {1, 3}
        self.is_free_turn = True
        self.jiefeng = False
        self.winning_team = 0
        self.is_game_over = False
        self.upgrade_amount = 0
        # 手牌排序
        for player in self.players:
            player.hand = self.sort_cards(player.hand)


    def log(self, message):
        """记录游戏日志"""
        st.session_state.game_log.append(message)
        if len(st.session_state.game_log) > 100:  # 限制日志长度
            st.session_state.game_log.pop(0)

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def map_cards_to_action(self, cards, M, level_rank):
        """
        从实际出过的牌中（带花色），判断其结构动作（含同花顺识别）。
        """
        point_count = defaultdict(int)
        suits = set()
        if not cards:
            cards = []
        for card in cards:
            for rank in RANKS + ['小王', '大王']:
                if rank in card:
                    raw_point = CARD_RANKS[rank]
                    logic_point = 15 if raw_point == level_rank else raw_point
                    point_count[logic_point] += 1
                    break
            # 提取花色
            for s in SUITS:
                if card.startswith(s):
                    suits.add(s)
                    break

        # 构建点数序列（带重复）
        logic_points = []
        for pt, count in sorted(point_count.items()):
            logic_points.extend([pt] * count)

        # 🔍 同花顺检测
        if len(cards) == 5 and len(point_count) == 5:
            sorted_points = sorted(point_count.keys())
            if all(sorted_points[i + 1] - sorted_points[i] == 1 for i in range(4)):
                if len(suits) == 1:
                    # 是同花顺 → 去 M 中找类型为 straight_flush
                    for action in M:
                        if action['type'] == 'flush_rocket' and sorted(action['points']) == sorted_points:
                            return action

        # 🔁 普通结构匹配
        for action in M:
            if sorted(action['points']) == sorted(logic_points):
                return action

        return None

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
            result = self.ai_play(player)
        # **记录最近 5 轮历史**
        if self.current_player == 0:
            round_history = [self.recent_actions[i] for i in range(4)]
            self.history.append(round_history)
            self.recent_actions = [['None'], ['None'], ['None'], ['None']]
            '''
            if len(self.history) > 20:
                self.history.pop(0)
            '''
        return result

    def get_possible_moves(self, player_hand):
        """获取所有可能的合法出牌，包括顺子（5 张）、连对（aabbcc）、钢板（aaabbb）"""

        possible_moves = []
        hand_points = [self.rules.get_rank(card) for card in player_hand]  # 仅点数（去掉花色）
        hand_counter = Counter(hand_points)  # 统计点数出现次数
        unique_points = sorted(set(hand_points))  # 仅保留唯一点数，排序

        # 1. **原逻辑（单张、对子、三条、炸弹等）**
        for size in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        # 2. **检查顺子（固定 5 张）**
        for i in range(len(unique_points) - 4):  # 只找长度=5 的顺子
            seq = unique_points[i:i + 5]
            if self.rules._is_consecutive(seq) and 15 not in seq:  # 不能有大小王
                move = self._map_back_to_suit(seq, player_hand)  # 还原带花色的牌
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        # 3. **检查连对（aabbcc）**
        for i in range(len(unique_points) - 2):  # 只找 3 组对子
            seq = unique_points[i:i + 3]
            if all(hand_counter[p] >= 2 for p in seq):  # 每张至少两张
                move = self._map_back_to_suit(seq, player_hand, count=2)  # 每点数取 2 张
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        # 4. **检查钢板（aaabbb）**
        for i in range(len(unique_points) - 1):  # 只找 2 组三张
            seq = unique_points[i:i + 2]
            if all(hand_counter[p] >= 3 for p in seq):  # 每张至少 3 张
                move = self._map_back_to_suit(seq, player_hand, count=3)  # 每点数取 3 张
                if self.rules.can_beat(self.last_play, move):
                    possible_moves.append(move)

        return possible_moves

    def _map_back_to_suit(self, seq, sorted_hand, count=1):
        """从手牌映射回带花色的牌"""
        move = []
        hand_copy = sorted_hand[:]  # 复制手牌
        for p in seq:
            for _ in range(count):  # 取 count 张
                for card in hand_copy:
                    if self.rules.get_rank(card) == p:
                        move.append(card)
                        hand_copy.remove(card)
                        break
        return move

    def can_beat(self, curr_action, prev_action):
        """
        判断结构动作 curr_action 是否能压过 prev_action
        """
        # 如果没人出牌，当前动作永远可以出
        if prev_action["type"] == "None":
            if curr_action["type"] == "None":
                return False
            else:
                return True

        curr_type = curr_action["type"]
        prev_type = prev_action["type"]

        # 炸弹类型（根据牌力表）
        bomb_power = {
            "joker_bomb": 6,
            "8_bomb": 5,
            "7_bomb": 4,
            "6_bomb": 3,
            "flush_rocket": 2,
            "5_bomb": 1,
            "4_bomb": 0
        }

        is_curr_bomb = curr_type in bomb_power
        is_prev_bomb = prev_type in bomb_power

        # ✅ 炸弹能压非炸弹
        if is_curr_bomb and not is_prev_bomb:
            return True
        if not is_curr_bomb and is_prev_bomb:
            return False

        # ✅ 两个都是炸弹 → 比炸弹牌力 → 再比 logic_point
        if is_curr_bomb and is_prev_bomb:
            if bomb_power[curr_type] > bomb_power[prev_type]:
                return True
            elif bomb_power[curr_type] < bomb_power[prev_type]:
                return False
            else:  # 相同牌力 → 比点数
                return curr_action["logic_point"] > prev_action["logic_point"]

        # ✅ 非炸弹时，牌型必须相同才可比
        if curr_type != prev_type:
            return False

        # ✅ 非炸弹，牌型相同 → 比 logic_point
        return curr_action["logic_point"] > prev_action["logic_point"]

    def get_valid_action_mask(self, hand, M, level_rank, last_action):
        """
        返回 mask 向量，标记每个结构动作在当前手牌下是否合法。
        如果 last_action 为 None，则为主动出牌，可出任意合法牌型；
        否则为跟牌回合，只能出能压过 last_action 的合法牌。
        """
        mask = np.zeros(len(M), dtype=np.float32)
        if not last_action:
            last_action = []
        last_action = self.map_cards_to_action(last_action, M, level_rank)
        for action in M:
            action_id = action['id']
            combos = enumerate_colorful_actions(action, hand, level_rank)
            if not combos:
                continue  # 当前手牌无法组成该结构

            if last_action is None:
                # 主动出牌：只要能组成即可
                mask[action_id] = 1.0
            else:
                # 跟牌出牌：还要能压上上家
                if self.can_beat(action, last_action):
                    mask[action_id] = 1.0
        if not self.is_free_turn:
            # 永远允许出 “None” 结构（pass）
            for action in M:
                if action['type'] == 'None':
                    mask[action['id']] = 1.0
                    break

        return mask

    def ai_play(self, player):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        # **如果玩家已经打完，仍然记录一个空列表，然后跳过**
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            self.current_player = (self.current_player + 1) % 4

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
            chosen_move = random.choice(possible_moves)  # 随机选择一个合法的牌型
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
                    player_hand.remove(card)
                self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
                self.recent_actions[self.current_player] = list(chosen_move)  # 记录出牌
                self.jiefeng = False
                if not player_hand:  # 玩家出完牌
                    self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                    self.ranking.append(self.current_player)
                    if len(self.ranking) <= 2:
                        self.jiefeng = True

                self.pass_count = 0
                if not player_hand:
                    self.pass_count -= 1

                if self.is_free_turn:
                    self.is_free_turn = False
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def actor_play(self, player):
        # 1. 模型推理
        state = self._get_obs()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(self.get_valid_action_mask(player.hand, M, self.active_level, self.last_play)).unsqueeze(0)
        probs = actor(state_tensor, mask)
        action_id = torch.multinomial(probs, 1).item()
        action_struct = M_id_dict[action_id]
        # 2. 枚举所有合法出牌组合（带花色）
        combos = enumerate_colorful_actions(action_struct, player.hand, self.active_level)
        if combos:
            chosen_move = random.choice(combos)
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
                    self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
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
        """用户出牌逻辑（Streamlit版本）"""
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []
            self.current_player = (self.current_player + 1) % 4
            return self.check_game_over()

        # 显示AI建议
        suggestions = self.get_ai_suggestions()
        with st.expander("AI建议"):
            for sug in suggestions:
                st.write(sug)

        # 显示手牌并获取选择
        selected_cards = self.show_user_hand()

        # 出牌和Pass按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("出牌", disabled=not selected_cards):
                if self.validate_and_play(selected_cards, player):
                    return self.advance_turn()

        with col2:
            if st.button("Pass", disabled=self.is_free_turn):
                self.log(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                self.recent_actions[self.current_player] = ['Pass']
                return self.advance_turn()

        return False

    def validate_and_play(self, selected_cards, player):
        """验证并执行出牌"""
        # 检查牌是否在手牌中
        if not all(card in player.hand for card in selected_cards):
            st.error("❌ 你的输入无效，请确保牌在你的手牌中！")
            return False

        # 检查牌是否合法
        if not self.rules.is_valid_play(selected_cards):
            st.error("❌ 你的出牌不符合规则，请重新选择！")
            return False

        last_action = self.map_cards_to_action(self.last_play, M, self.active_level)
        chosen = self.map_cards_to_action(selected_cards, M, self.active_level)

        # 检查是否能压过上一手牌
        if not self.can_beat(chosen, last_action):
            st.error("❌ 你的牌无法压过上一手牌，请重新选择！")
            return False

        # 成功出牌
        for card in selected_cards:
            player.played_cards.append(card)
            player.hand.remove(card)
        self.last_play = selected_cards
        self.last_player = self.current_player
        self.recent_actions[self.current_player] = list(selected_cards)
        self.jiefeng = False
        self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(selected_cards)}")

        # 如果手牌为空，玩家出完所有牌
        if not player.hand:
            self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
            self.ranking.append(self.current_player)
            if len(self.ranking) <= 2:
                self.jiefeng = True

        self.pass_count = 0
        if not player.hand:
            self.pass_count -= 1
        if self.is_free_turn:
            self.is_free_turn = False

        return True
    def get_ai_suggestions(self):
        """
        Generates top 3 AI move suggestions for the current player (intended for user player).
        Returns a list of formatted strings describing the suggestions.
        """
        if self.current_player != self.user_player:
            return ["AI suggestions only available on your turn."]

        player = self.players[self.current_player]
        suggestions = []

        try:
            state = self._get_obs()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(
                self.get_valid_action_mask(player.hand, M, self.active_level, self.last_play)).unsqueeze(0)

            global actor  # Assuming actor is loaded globally
            with torch.no_grad():
                all_probs = actor(state_tensor, mask)

            top_k_orig_probs, top_k_indices = torch.topk(all_probs, k=3, dim=-1)

            # Apply Softmax to ONLY the top K probabilities for relative comparison
            valid_top_k_probs = top_k_orig_probs[top_k_orig_probs > 0]
            if valid_top_k_probs.numel() > 0:
                normalized_top_k_probs_tensor = F.softmax(valid_top_k_probs, dim=-1)
                normalized_top_k_probs = torch.zeros_like(top_k_orig_probs)
                normalized_top_k_probs[top_k_orig_probs > 0] = normalized_top_k_probs_tensor
            else:
                normalized_top_k_probs = torch.zeros_like(top_k_orig_probs)

            for i in range(top_k_indices.size(1)):
                action_id = top_k_indices[0, i].item()
                normalized_prob = normalized_top_k_probs[0, i].item()

                if top_k_orig_probs[0, i].item() > 0:  # Check if it was a valid move initially
                    action_struct = M_id_dict.get(action_id)
                    if action_struct:
                        action_desc = action_struct.get('name', action_struct.get('type', f'动作ID {action_id}'))
                        points_str = ""
                        if 'points' in action_struct and action_struct['points']:
                            points_str = f" (点数: {action_struct['points']})"
                        if action_struct.get('type') == 'None':
                            action_desc = "Pass (不出)"
                            points_str = ""
                        suggestions.append(f"建议 {i + 1}: {action_desc}{points_str} - 相对概率: {normalized_prob:.2%}")
                    else:
                        suggestions.append(f"建议 {i + 1}: 未知动作 ID {action_id} - 相对概率: {normalized_prob:.2%}")
                else:
                    suggestions.append(f"建议 {i + 1}: (无有效动作)")

        except Exception as e:
            suggestions.append(f"Error getting suggestions: {e}")

        # Ensure "Pass" suggestion is included if applicable
        can_pass = not self.is_free_turn
        pass_in_suggestions = any("Pass" in s for s in suggestions)
        if can_pass and not pass_in_suggestions:
            # Check if Pass action exists and is valid according to the mask
            pass_action_id = -1
            for action_id, action_struct in M_id_dict.items():
                if action_struct.get('type') == 'None':
                    pass_action_id = action_id
                    break
            if pass_action_id != -1 and mask[0, pass_action_id].item() > 0:
                # If pass is valid but not in top-k, add it manually (or adjust top-k logic)
                # For simplicity, just noting it might be missing from top 3
                pass  # Or potentially add: suggestions.append("建议: Pass (不出) - [概率未在Top3]")
            elif not can_pass:
                # Remove pass suggestion if it's a free turn
                suggestions = [s for s in suggestions if "Pass" not in s]

        if not suggestions:
            suggestions.append("无可用建议。")

        return suggestions

    def check_game_over(self):
        """检查游戏是否结束"""
        # **如果有 2 个人出完牌，并且他们是同一队伍，游戏立即结束**
        if len(self.ranking) >= 2:
            first_player, second_player = self.ranking[0], self.ranking[1]
            if (first_player in self.team_1 and second_player in self.team_1) or (
                    first_player in self.team_2 and second_player in self.team_2):
                self.ranking.extend(i for i in range(4) if i not in self.ranking)  # 剩下的按出牌顺序补全
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
        """升级级牌"""
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
        self.upgrade_amount=upgrade_amount

        self.log(f"\n🏆 {winning_team} 号队伍获胜！得 {upgrade_amount} 分")
        # 显示最终排名
        ranks = ["头游", "二游", "三游", "末游"]
        for i, player in enumerate(self.ranking):
            self.log(f"{ranks[i]}：玩家 {player + 1}")

    def advance_turn(self):
        """推进到下一个回合"""
        player = self.players[self.current_player]
        player.last_played_cards = self.recent_actions[self.current_player]
        self.current_player = (self.current_player + 1) % 4

        # 自动进行AI回合
        while self.current_player != self.user_player and not self.is_game_over:
            if self.play_turn():  # AI出牌
                self.is_game_over = True
                break

        return self.is_game_over

    def play_game(self):
        """执行游戏主循环"""
        self.log(f"\n🎮 游戏开始！当前级牌：{RANKS[self.active_level - 2]}")

        while not self.is_game_over:
            if self.play_turn():
                if self.current_player != 0:
                    round_history = [self.recent_actions[i] for i in range(4)]
                    self.history.append(round_history)
                break
            time.sleep(1)  # 添加延迟使游戏节奏更自然

    def show_user_hand(self):
        """显示用户手牌"""
        if self.user_player is not None:
            sorted_hand = self.players[self.user_player].hand
            st.session_state.user_hand = " ".join(sorted_hand)

            # 创建卡片选择器
            selected = []
            cols = st.columns(5)
            for i, card in enumerate(sorted_hand):
                with cols[i % 5]:
                    if st.checkbox(card, key=f"card_{i}"):
                        selected.append(card)
            return selected

    def _get_obs(self):
        """
        构造状态向量，总共 3049 维
        """
        obs = np.zeros(3049)

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

        # 6️⃣ 最近 20 步动作历史 (108 * 20 = 2160)
        HISTORY_LEN = 20
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

        # 7️⃣ 状态向量 (9)
        obs[offset:offset + 3] = self.compute_coop_status()
        obs[offset + 3:offset + 6] = self.compute_dwarf_status()
        obs[offset + 6:offset + 9] = self.compute_assist_status()
        offset += 9

        assert offset == 3049, f"⚠️ offset 计算错误: 预期 3049, 实际 {offset}"
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


def main():
    st.title("🎴 AI掼蛋游戏")

    # 初始化session state（新增游戏状态跟踪）
    if 'game' not in st.session_state:
        st.session_state.game = None
        st.session_state.game_over = False
        st.session_state.auto_play = False

    if 'game_log' not in st.session_state:
        st.session_state.game_log = []

    # 游戏控制侧边栏（修改为使用form避免重复提交）
    with st.sidebar:
        with st.form("game_setup"):
            st.header("游戏设置")
            user_position = st.selectbox("选择你的位置", [1, 2, 3, 4], index=0)
            level_card = st.selectbox("选择级牌", RANKS, index=0)
            submitted = st.form_submit_button("开始游戏")

            if submitted:
                level_index = RANKS.index(level_card) + 2
                st.session_state.game = GuandanGame(user_player=user_position, active_level=level_index)
                st.session_state.game_log = []
                st.session_state.game_over = False
                st.rerun()  # 使用正式版rerun替代experimental_rerun

    # 游戏主界面（添加容器避免闪烁）
    game_container = st.container()

    if st.session_state.game and not st.session_state.game_over:
        with game_container:
            # 显示游戏状态
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    player = st.session_state.game.players[i]
                    st.subheader(f"玩家 {i + 1}")
                    st.write(f"剩余牌数: {len(player.hand)}")
                    if i == st.session_state.game.user_player:
                        st.write("👤 你")
                    if i in st.session_state.game.ranking:
                        st.success(f"排名: {st.session_state.game.ranking.index(i) + 1}")

            # 显示当前出牌
            if st.session_state.game.last_play:
                st.subheader("场上最新出牌")
                st.code(" ".join(st.session_state.game.last_play))

            # 游戏日志（限制显示数量）
            st.subheader("游戏日志")
            log_display = st.empty()
            log_text = "\n".join(st.session_state.game_log[-10:])
            log_display.text_area("日志", value=log_text, height=150, disabled=True)

            # 玩家回合处理
            if st.session_state.game.current_player == st.session_state.game.user_player:
                st.session_state.game.user_play(st.session_state.game.players[st.session_state.game.user_player])
            else:
                if st.button("AI自动出牌"):
                    st.session_state.auto_play = True

                if st.session_state.auto_play:
                    st.session_state.game.play_turn()
                    if st.session_state.game.is_game_over:
                        st.session_state.game_over = True
                        st.rerun()
                    else:
                        time.sleep(0.5)
                        st.rerun()

    elif st.session_state.game_over:
        with game_container:
            st.balloons()
            st.success("游戏结束！")
            ranks = ["头游", "二游", "三游", "末游"]
            for i, player in enumerate(st.session_state.game.ranking):
                st.write(f"{ranks[i]}：玩家 {player + 1}")

            if st.button("重新开始"):
                st.session_state.game = None
                st.session_state.game_over = False
                st.rerun()

    else:
        st.info("请点击侧边栏的开始游戏按钮开始新游戏")

if __name__ == "__main__":
    main()

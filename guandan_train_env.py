import random
from collections import Counter
import json
import numpy as np
with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)
# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}
# 定义牌型
SUITS = ['黑桃', '红桃', '梅花', '方块']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

# 创建两副牌
def create_deck():
    deck = []
    for _ in range(2):  # 两副牌
        for suit in SUITS:
            for rank in RANKS:
                card = f"{suit}{rank}"
                deck.append(card)
        # 添加大小王
        deck.append('小王')
        deck.append('大王')
    return deck

# 洗牌
def shuffle_deck(deck):
    random.shuffle(deck)
    return deck

# 发牌
def deal_cards(deck):
    players = [[], [], [], []]  # 4个玩家
    for i in range(len(deck)):
        players[i % 4].append(deck[i])
    return players


# 获取级牌
def get_level_card(current_round):
    # 级牌通常是当前局数的数字（如第2局的级牌是2）
    return str(current_round)

# 手牌排序（从大到小）
def sort_cards(cards, level_card=None):
    def get_card_value(card):
        if card == '大王':
            return (CARD_RANKS['大王'], 0)
        if card == '小王':
            return (CARD_RANKS['小王'], 0)
        rank = card[2:] if len(card) > 2 else card[2]  # 处理10以上的牌
        if level_card and rank == level_card:
            return (CARD_RANKS['A'] + 1, SUITS.index(card[:2]))  # 级牌仅小于大小王
        return (CARD_RANKS.get(rank, 0), SUITS.index(card[:2]))

    return sorted(cards, key=get_card_value, reverse=True)

class Rules:
    def __init__(self, level_card=None):
        self.level_card = level_card  # 级牌
        self.CARD_RANKS = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
            '小王': 16, '大王': 17
        }
        self.RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def is_valid_play(self, cards):
        """判断出牌是否合法"""
        if not cards:
            return False
        length = len(cards)

        if length == 1:
            return True  # 单张
        if length == 2:
            return self.is_pair(cards)  # 对子
        if length == 3:
            return self.is_triple(cards)  # 三同张
        if length == 4:
            return self.is_king_bomb(cards) or self.is_bomb(cards)  # 天王炸 or 4 炸
        if length == 5:
            return self.is_straight(cards) or self.is_flush_straight(cards) or self.is_three_with_two(
                cards) or self.is_bomb(cards)  # 顺子 / 同花顺 / 三带二
        if length == 6:
            return self.is_triple_pair(cards) or self.is_triple_consecutive(cards) or self.is_bomb(cards)  # 连对（木板） / 钢板
        if 6 < length <= 8:
            return self.is_bomb(cards)
        return False  # 其他情况不合法

    def is_pair(self, cards):
        """对子"""
        return len(cards) == 2 and self.get_rank(cards[0]) == self.get_rank(cards[1])

    def is_triple(self, cards):
        """三同张（三不带）"""
        return len(cards) == 3 and len(set(self.get_rank(card) for card in cards)) == 1

    def is_three_with_two(self, cards):
        """三带二"""
        if len(cards) != 5:
            return False
        counts = Counter(self.get_rank(card) for card in cards)
        return 3 in counts.values() and 2 in counts.values()

    def is_triple_pair(self, cards):
        """连对（木板），如 556677"""
        if len(cards) != 6:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = [self.get_rank(card, as_one=False) for card in cards]
        ranks_as_one = [self.get_rank(card, as_one=True) for card in cards]

        # 统计点数出现次数
        counts = Counter(ranks)
        counts_as_one = Counter(ranks_as_one)

        # 获取所有 **点数为 2 的对子**
        pairs = sorted([rank for rank, count in counts.items() if count == 2])
        pairs_as_one = sorted([rank for rank, count in counts_as_one.items() if count == 2])

        # 必须有 3 组对子，并且它们的点数是连续的
        return (len(pairs) == 3 and self._is_consecutive(pairs)) or \
            (len(pairs_as_one) == 3 and self._is_consecutive(pairs_as_one))

    def is_triple_consecutive(self, cards):
        """三同连张（钢板），如 555666"""
        if len(cards) != 6:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = [self.get_rank(card, as_one=False) for card in cards]
        ranks_as_one = [self.get_rank(card, as_one=True) for card in cards]

        # 统计点数出现次数
        counts = Counter(ranks)
        counts_as_one = Counter(ranks_as_one)

        # 获取所有 **点数为 3 的三同张**
        triples = sorted([rank for rank, count in counts.items() if count == 3])
        triples_as_one = sorted([rank for rank, count in counts_as_one.items() if count == 3])

        # 必须有 2 组三同张，并且它们的点数是连续的
        return (len(triples) == 2 and self._is_consecutive(triples)) or \
            (len(triples_as_one) == 2 and self._is_consecutive(triples_as_one))

    def is_straight(self, cards):
        """顺子（必须 5 张，A 可作为 1 或 14）"""
        if len(cards) != 5:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
        ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)

        # 检查 A=1 或 A=14 的情况
        return self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one)

    def is_flush_straight(self, cards):
        """同花顺（火箭），如 ♠10JQKA"""
        if len(cards) != 5:
            return False

        # 获取所有牌的点数（去掉花色）
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
        ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)

        # 获取所有牌的花色
        suits = {card[:2] for card in cards}

        # 需要 **同一花色** 且 **顺序正确**
        return len(suits) == 1 and (self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one))

    def is_bomb(self, cards):
        """炸弹（5 张及以上的相同牌 or 4 张相同牌）"""
        if len(cards) < 4:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) == 1


    def is_king_bomb(self, cards):
        """四大天王（天王炸）"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def get_rank(self, card, as_one=False):
        """获取牌的点数，支持 A=1"""
        if card in ['小王', '大王']:
            return CARD_RANKS[card]

        rank = card[2:] if len(card) > 2 else card[2]  # 解析点数

        # **只检查当前局的级牌**
        if rank == RANKS[self.level_card - 2]:
            return CARD_RANKS['A'] + 1  # 级牌比 A 还大

        if as_one and rank == 'A':
            return 1  # A 作为 1

        return CARD_RANKS.get(rank, 0)

    def _is_consecutive(self, ranks):
        """判断是否为连续数字序列"""
        return all(ranks[i] == ranks[i - 1] + 1 for i in range(1, len(ranks)))

    def can_beat(self, previous_play, current_play):
        """判断当前出牌是否能压过上家"""
        if not self.is_valid_play(current_play):
            return False
        if not previous_play:
            return True  # 没人出牌，可以随便出

        prev_type = self.get_play_type(previous_play)
        curr_type = self.get_play_type(current_play)


        # **修正炸弹牌力顺序**
        bomb_order = ['天王炸', '8炸', '7炸', '6炸', '同花顺', '5炸', '4炸']

        # **炸弹能压制非炸弹**
        if curr_type in bomb_order and prev_type not in bomb_order:
            return True
        if prev_type in bomb_order and curr_type in bomb_order:
            return bomb_order.index(curr_type) < bomb_order.index(prev_type)

        # **牌型必须相同才能比较**
        if prev_type != curr_type:
            return False

        return self.get_play_value(current_play) > self.get_play_value(previous_play)

    def get_play_type(self, cards):
        """获取牌型"""
        if self.is_king_bomb(cards):
            return '天王炸'
        if self.is_flush_straight(cards):
            return '同花顺'
        if self.is_bomb(cards):
            size = len(cards)
            if size == 4:
                return '4炸'
            elif size == 5:
                return '5炸'
            elif size == 6:
                return '6炸'
            elif size == 7:
                return '7炸'
            elif size == 8:
                return '8炸'
        if self.is_triple_consecutive(cards):
            return '钢板'
        if self.is_triple_pair(cards):
            return '木板'
        if self.is_three_with_two(cards):
            return '三带二'
        if self.is_triple(cards):
            return '三同张'
        if self.is_straight(cards):
            return '顺子'
        if self.is_pair(cards):
            return '对子'
        if len(cards) == 1:
            return '单牌'
        return '非法牌型'

    def get_play_value(self, cards):
        """获取牌点数"""
        ranks = [self.get_rank(card) for card in cards]
        return max(ranks)




# 2025/4/17 10:22
from collections import defaultdict
from itertools import combinations, product
RANK_STR = {v: k for k, v in CARD_RANKS.items()}

def parse_hand(hand):
    """将手牌转换为点数到花色牌组的映射"""
    point_to_cards = defaultdict(list)
    for card in hand:
        for rank in RANKS + ['小王', '大王']:
            if rank in card:
                point = CARD_RANKS[rank]
                point_to_cards[point].append(card)
                break
    return point_to_cards

def find_combinations(points, point_to_cards):
    """递归回溯地找出所有不重复使用牌的组合（通过索引区分同名牌）"""
    results = []

    card_pool = []  # (point, card) 的列表
    card_to_idx = {}  # point ➔ 对应的全局 idx 列表
    idx = 0
    for p, cards in point_to_cards.items():
        for c in cards:
            card_pool.append((p, c))
            card_to_idx.setdefault(p, []).append(idx)
            idx += 1  # 全局编号递增

    def backtrack(index, path, used_idx):
        if index == len(points):
            results.append([card_pool[i][1] for i in path])  # 只保留牌面
            return
        p = points[index]
        available = [i for i in card_to_idx.get(p, []) if i not in used_idx]
        if not available:
            return  # ❌ 找不到需要的牌直接剪枝
        for i in available:
            used_idx.add(i)
            path.append(i)
            backtrack(index + 1, path, used_idx)
            path.pop()
            used_idx.remove(i)

    backtrack(0, [], set())
    return results

def parse_hand_with_level(hand, level_rank: int):
    """
    将手牌转换为 point_to_cards 映射：
    - 如果是级牌（点数 == level_rank），在逻辑上映射为 15 点
    """
    point_to_cards = defaultdict(list)
    for card in hand:
        for rank in RANKS + ['小王', '大王']:
            if rank in card:
                raw_point = CARD_RANKS[rank]
                logic_point = 15 if raw_point == level_rank else raw_point
                point_to_cards[logic_point].append(card)
                break
    return point_to_cards

def group_points(points):
    counts = defaultdict(int)
    for p in points:
        counts[p] += 1
    return dict(counts)

def match_structured_action(points, point_to_cards):
    grouped = group_points(points)
    group_by_count = defaultdict(list)
    for pt, cnt in grouped.items():
        group_by_count[cnt].append(pt)

    all_combos = []

    # ✅ 三带二：3+2
    if set(grouped.values()) == {3, 2} and len(grouped) == 2:
        triples = group_by_count[3]
        pairs = group_by_count[2]
        for triple_point in triples:
            for triple_cards in combinations(point_to_cards.get(triple_point, []), 3):
                for pair_point in pairs:
                    if pair_point == triple_point:
                        continue
                    for pair_cards in combinations(point_to_cards.get(pair_point, []), 2):
                        all_combos.append(list(triple_cards) + list(pair_cards))

    # ✅ 连对：3个连续点数，每个2张（如 334455）
    elif all(cnt == 2 for cnt in grouped.values()) and len(grouped) >= 3:
        seq = sorted(grouped.keys())
        if all(seq[i+1] - seq[i] == 1 for i in range(len(seq) - 1)):
            pair_options = []
            for pt in seq:
                pair_options.append(list(combinations(point_to_cards.get(pt, []), 2)))
            for pairs in product(*pair_options):
                combo = [card for pair in pairs for card in pair]
                all_combos.append(combo)

    # ✅ 钢板：2个连续点数，每个3张（如 555666）
    elif all(cnt == 3 for cnt in grouped.values()) and len(grouped) == 2:
        seq = sorted(grouped.keys())
        if seq[1] - seq[0] == 1:
            triple_options = []
            for pt in seq:
                triple_options.append(list(combinations(point_to_cards.get(pt, []), 3)))
            for triples in product(*triple_options):
                combo = [card for trip in triples for card in trip]
                all_combos.append(combo)

    return all_combos

def enumerate_colorful_actions(action, hand, level_rank: int):
    point_to_cards = parse_hand_with_level(hand, level_rank)
    raw_combos = []

    # 特判结构型牌型（三带二）
    structured_combos = match_structured_action(action['points'], point_to_cards)
    raw_combos.extend(structured_combos)

    # 普通动作（顺子/对子等）
    if not structured_combos:
        raw_combos = find_combinations(action['points'], point_to_cards)

    if action['type'] == 'flush_rocket':
        filtered_combos = []
        for combo in raw_combos:
            suits = [card[:2] for card in combo]
            if all(s == suits[0] for s in suits):  # 花色必须完全一致
                filtered_combos.append(combo)
        raw_combos = filtered_combos

    # 去重
    seen = set()
    unique_combos = []
    for combo in raw_combos:
        key = frozenset(combo)
        if key not in seen:
            seen.add(key)
            unique_combos.append(combo)
    return unique_combos

def build_card_index_map():
    index_map = {}
    idx = 0
    for copy in range(2):
        for suit in ['黑桃', '红桃', '梅花', '方块']:
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
                card = f"{suit}{rank}"
                if card not in index_map:
                    index_map[card] = []
                index_map[card].append(idx)
                idx += 1
    for copy in range(2):
        index_map.setdefault('小王', []).append(idx)
        idx += 1
        index_map.setdefault('大王', []).append(idx)
        idx += 1
    return index_map

def encode_hand_108(hand):
    card_map = build_card_index_map()
    obs = np.zeros(108)
    if hand == ['Pass'] or hand == ['None'] or not hand:
        hand = []

    card_count = {}
    for card in hand:
        card_count[card] = card_count.get(card, 0) + 1

    for card, count in card_count.items():
        indices = card_map.get(card, [])
        for i in range(min(count, len(indices))):
            obs[indices[i]] = 1.0
    return obs


class Player:
    def __init__(self, hand):
        """
        程序里的玩家是从0开始的，输出时会+1
        """
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌
        self.last_played_cards = []

class GuandanGame:
    def __init__(self, user_player: int, active_level=None, verbose=True, print_history=False,test=False, model_path="models/show2.pth",sug_len=3):
        # **两队各自的级牌**
        self.print_history = print_history
        self.active_level = active_level if active_level else random.choice(range(2, 15))
        # 历史记录，记录最近 20 轮的出牌情况（每轮包含 4 个玩家的出牌）
        self.history = []
        # **只传当前局的有效级牌**
        self.rules = Rules(self.active_level)
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]  # 发牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = []  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        if not isinstance(user_player, int) or user_player not in {1, 2, 3, 4}:
            raise ValueError("user_player 必须是 1-4 的整数")
        self.user_player = int(user_player - 1) if user_player else None  # 转换为索引（0~3）
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
        self.test=test
        self.model_path = model_path
        self.R = RANKS + [RANKS[self.active_level-2]] + ['小王', '大王']
        self.sug_len = sug_len

        # **手牌排序**
        for player in self.players:
            player.hand = self.sort_cards(player.hand)

    def point_to_card(self,point) -> [str, list]:
        if isinstance(point, list):
            return [str(self.R[int(p) - 2]) for p in point]
        else:
            return str(self.R[point - 2])

    def log(self, message):
        """控制是否打印消息"""
        if self.verbose:
            print(message)

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

    def maybe_reset_turn(self):
        """计算当前仍有手牌的玩家数、处理接风等复杂逻辑"""
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
        # **记录最近 5 轮历史**
        if self.current_player == 0:
            round_history = [self.recent_actions[i] for i in range(4)]
            self.history.append(round_history)
            self.recent_actions = [['None'], ['None'], ['None'], ['None']]
            '''
            if len(self.history) > 20:
                self.history.pop(0)
            '''

    def play_turn(self):
        """执行当前玩家的回合"""
        player = self.players[self.current_player]  # 获取当前玩家对象
        # TODO: 添加选座位接口
        if self.user_player == self.current_player:
            result = self.user_play(player)
        else:
            result = self.ai_play(player)

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


    def user_play(self, player):
        """用户出牌逻辑"""
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            self.current_player = (self.current_player + 1) % 4
            return self.check_game_over()

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
            if  not self.can_beat(chosen,last_action):
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
                print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
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

    def play_game(self):
        """执行一整局游戏"""
        self.log(f"\n🎮 游戏开始！当前级牌：{RANKS[self.active_level - 2]}")

        while True:
            if self.play_turn():
                if self.current_player != 0:
                    round_history = [self.recent_actions[i] for i in range(4)]
                    self.history.append(round_history)
                if self.print_history:
                    for i in range(len(self.history)):
                        self.log(self.history[i])
                break

    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.players[self.user_player].hand
        print("\n你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"场上最新出牌：{' '.join(self.last_play)}\n")
    # DONE: 检查不同座位能否正常工作
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

    def compute_reward(self):
        """计算当前的奖励"""
        if self.check_game_over():
            # 如果游戏结束，给胜利队伍正奖励，失败队伍负奖励
            return 100 if self.current_player in self.winning_team else -100

        # **鼓励 AI 先出完手牌**
        hand_size = len(self.players[self.current_player].hand)
        return -hand_size  # 手牌越少，奖励越高

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

    def submit_user_move(self, selected_cards):
        """前端提交出牌：selected_cards为list[str]，如 ['红桃3', '黑桃3'] 或 []"""
        if self.is_game_over:
            return {"error": "游戏已结束"}

        player = self.players[self.user_player]

        if selected_cards == []:  # 选择 PASS
            if self.is_free_turn:
                return {"error": "自由回合必须出牌"}
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']
        else:
            if not all(card in player.hand for card in selected_cards):
                return {"error": "出牌不在手牌中"}

            if not self.rules.is_valid_play(selected_cards):
                return {"error": "出牌不合法"}

            if not self.can_beat(self.map_cards_to_action(selected_cards, M, self.active_level),
                                 self.map_cards_to_action(self.last_play, M, self.active_level)):
                return {"error": "不能压过上家"}

            for card in selected_cards:
                player.hand.remove(card)
                player.played_cards.append(card)

            self.last_play = selected_cards
            self.last_player = self.current_player
            self.recent_actions[self.current_player] = selected_cards
            if not player.hand:
                self.ranking.append(self.current_player)
                if len(self.ranking) <= 2:
                    self.jiefeng = True
            self.pass_count = 0
            if self.is_free_turn:
                self.is_free_turn = False

        self.current_player = (self.current_player + 1) % 4
        self.maybe_reset_turn()
        return {"success": True, "game_over": self.check_game_over()}

    def step(self):
        """推进一步（仅用于非用户玩家），返回字典说明状态"""
        if self.is_game_over:
            return {"game_over": True}

        if self.current_player == self.user_player:
            self.user_play(self.players[self.user_player])
            self.maybe_reset_turn()
            return {"waiting_for_user": True}

        # 处理 AI 或其他自动玩家的出牌
        self.play_turn()
        self.maybe_reset_turn()

        # 如果刚好出完最后一张牌并结束
        if self.is_game_over:
            return {"game_over": True}

        # 如果下一个轮到用户，告诉前端等待
        if self.current_player == self.user_player:
            return {"waiting_for_user": True}

        # 否则仍轮到 AI，下次前端可继续调用 step
        return {"next_step_needed": True}

    def get_player_statuses(self):
        """
        返回每位玩家的状态，用于前端显示：
        [
            {'id': 1, 'hand_size': 15, 'last_play': ['红桃3', '黑桃3']},
            ...
        ]
        """
        result = []
        for i, player in enumerate(self.players):
            result.append({
                "id": i + 1,
                "hand_size": len(player.hand),
                "last_play": player.last_played_cards
            })
        return result

    def get_game_state(self):
        """获取游戏的完整可视状态字典，供前端展示"""
        return {
            "user_hand": self.players[self.user_player].hand,
            "last_play": self.last_play,
            "current_player": self.current_player,
            "history": self.history,
            "ranking": self.ranking,
            "is_game_over": self.is_game_over,
            "level_rank": self.active_level,
            "recent_actions": self.recent_actions
        }
if __name__ == "__main__":
    game = GuandanGame(user_player=1)
    game.play_game()

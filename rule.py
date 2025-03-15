from collections import Counter

# 定义牌型和优先级
CARD_TYPES = {
    '单牌': 1,
    '对子': 2,
    '三带': 3,
    '顺子': 4,
    '连对': 5,
    '飞机': 6,
    '炸弹': 7,
    '王炸': 8
}

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

class Rules:
    def __init__(self, level_card=None):
        self.level_card = level_card  # 级牌

    # 判断一手牌是否合法
    def is_valid_play(self, cards):
        if not cards:
            return False
        length = len(cards)
        if length == 1:
            return self.is_single(cards)  # 单牌
        if length == 2:
            return self.is_pair(cards) or self.is_rocket(cards)  # 对子或王炸
        if length == 3:
            return self.is_triple(cards)  # 三带
        if length == 4:
            return self.is_bomb(cards)  # 炸弹
        if length == 5:
            return self.is_straight(cards) or self.is_three_with_two(cards)  # 顺子或三带二
        if length >= 6:
            return self.is_straight(cards) or self.is_double_straight(cards) or \
                   self.is_airplane(cards) or self.is_bomb(cards)  # 顺子、连对、飞机、炸弹
        return False

    # 判断是否为单牌
    def is_single(self, cards):
        return len(cards) == 1

    # 判断是否为对子
    def is_pair(self, cards):
        if len(cards) != 2:
            return False
        rank1 = self.get_rank(cards[0])
        rank2 = self.get_rank(cards[1])
        return rank1 == rank2

    # 判断是否为三带
    def is_triple(self, cards):
        if len(cards) != 3:
            return False
        rank1 = self.get_rank(cards[0])
        rank2 = self.get_rank(cards[1])
        rank3 = self.get_rank(cards[2])
        return rank1 == rank2 == rank3

    # 判断是否为顺子
    def is_straight(self, cards):
        if len(cards) < 5:
            return False
        ranks = sorted([self.get_rank(card) for card in cards])
        # 处理A作为1的情况
        if ranks[-1] == 14:  # A
            ranks_alt = ranks[:-1] + [1]
            ranks_alt.sort()
            if self._is_consecutive(ranks_alt):
                return True
        return self._is_consecutive(ranks)

    # 判断是否为连对
    def is_double_straight(self, cards):
        if len(cards) < 6 or len(cards) % 2 != 0:
            return False
        ranks = sorted([self.get_rank(card) for card in cards])
        pairs = [ranks[i] for i in range(0, len(ranks), 2)]
        return self._is_consecutive(pairs)

    # 判断是否为飞机
    def is_airplane(self, cards):
        if len(cards) < 6 or len(cards) % 3 != 0:
            return False
        ranks = [self.get_rank(card) for card in cards]
        count = Counter(ranks)
        triples = [rank for rank, cnt in count.items() if cnt >= 3]
        triples.sort()
        return self._is_consecutive(triples)

    # 判断是否为三带二
    def is_three_with_two(self, cards):
        if len(cards) != 5:
            return False
        ranks = [self.get_rank(card) for card in cards]
        count = Counter(ranks)
        return any(cnt == 3 for cnt in count.values()) and any(cnt == 2 for cnt in count.values())

    # 判断是否为炸弹
    def is_bomb(self, cards):
        if len(cards) < 4:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) == 1

    # 判断是否为王炸
    def is_rocket(self, cards):
        return set(cards) == {'小王', '大王'}

    # 判断当前出牌是否能压制上家
    def can_beat(self, previous_play, current_play):
        if not self.is_valid_play(current_play):
            return False
        if not previous_play:
            return True  # 上家没出牌，可以直接出
        prev_type = self.get_play_type(previous_play)
        curr_type = self.get_play_type(current_play)
        if curr_type == '王炸':
            return True
        if curr_type == '炸弹' and prev_type != '炸弹' and prev_type != '王炸':
            return True
        if curr_type != prev_type:
            return False
        return self.get_play_value(current_play) > self.get_play_value(previous_play)

    # 获取出牌类型
    def get_play_type(self, cards):
        if self.is_rocket(cards):
            return '王炸'
        if self.is_bomb(cards):
            return '炸弹'
        if self.is_airplane(cards):
            return '飞机'
        if self.is_double_straight(cards):
            return '连对'
        if self.is_straight(cards):
            return '顺子'
        if self.is_three_with_two(cards):
            return '三带二'
        if self.is_triple(cards):
            return '三带'
        if self.is_pair(cards):
            return '对子'
        if self.is_single(cards):
            return '单牌'
        return '非法牌型'

    # 获取出牌的点数值
    def get_play_value(self, cards):
        if self.is_rocket(cards):
            return float('inf')
        if self.is_bomb(cards):
            return self.get_rank(cards[0]) * 100
        if self.is_airplane(cards):
            ranks = [self.get_rank(card) for card in cards]
            return max(ranks)
        if self.is_double_straight(cards):
            ranks = [self.get_rank(card) for card in cards]
            return max(ranks)
        if self.is_straight(cards):
            ranks = [self.get_rank(card) for card in cards]
            return max(ranks)
        if self.is_three_with_two(cards):
            ranks = [self.get_rank(card) for card in cards]
            return max(ranks)
        if self.is_triple(cards):
            return self.get_rank(cards[0])
        if self.is_pair(cards):
            return self.get_rank(cards[0])
        if self.is_single(cards):
            return self.get_rank(cards[0])
        return 0

    # 获取牌的点数（支持级牌）
    def get_rank(self, card):
        if card in ['小王', '大王']:
            return CARD_RANKS[card]  # 直接返回大小王的点数
        if self.level_card and self.level_card in card:
            return CARD_RANKS['A'] + 1  # 级牌仅小于大小王
        rank = card[2:] if len(card) > 2 else card[2]  # 处理10以上的牌
        return CARD_RANKS.get(rank, 0)

    # 判断是否为连续序列
    def _is_consecutive(self, ranks):
        for i in range(1, len(ranks)):
            if ranks[i] != ranks[i - 1] + 1:
                return False
        return True

if __name__ == "__main__":
    # 当前局数（假设为第2局）
    current_round = 2
    #rules = Rules(level_card=str(current_round))
    rules = Rules(level_card=None)

    # 测试牌型判断
    print(rules.is_valid_play(['黑桃2', '红桃2']))  # 对子
    print(rules.is_valid_play(['黑桃3', '红桃4', '方块5']))  # 非法牌型
    print(rules.is_valid_play(['黑桃3', '红桃4', '方块5', '梅花6', '黑桃7']))  # 顺子
    print(rules.is_valid_play(['小王', '大王']))  # 王炸

    # 测试出牌规则
    print(rules.can_beat(['黑桃2', '红桃2'], ['黑桃3', '红桃3']))  # True
    print(rules.can_beat(['黑桃3', '红桃3'], ['黑桃2', '红桃2']))  # False
    print(rules.can_beat(['黑桃2', '红桃2'], ['小王', '大王']))  # True
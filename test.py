from collections import Counter
from itertools import product

CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

class Rules:
    def __init__(self, level_card=None):
        self.level_card = level_card  # 级牌

    def has_wildcard(self, cards):
        """检查手牌中是否有逢人配"""
        return any('红桃' in card and self.level_card in card for card in cards)

    def get_possible_replacements(self, cards):
        """获取逢人配可以替换的合理点数"""
        possible_values = set()
        counter = Counter(self.get_rank(card) for card in cards if '红桃' not in card)

        # 如果是炸弹，逢人配只能变成已有的点数
        if len(cards) >= 4 and len(counter) == 1:
            return {next(iter(counter.keys()))}

            # 如果是对子或三带二，逢人配只能变成已有的对子/三张
        if len(cards) in [2, 3, 5]:
            for rank, count in counter.items():
                if count >= 1:
                    possible_values.add(rank)

        # 如果是顺子，逢人配只能变成顺子缺失的牌
        if len(cards) == 5:
            ranks = sorted(counter.keys())
            for i in range(min(ranks) - 1, max(ranks) + 2):
                if i not in ranks:
                    possible_values.add(i)

        # 确保逢人配不会变成离谱的点数
        return {r for r in possible_values if 2 <= r <= 14}

    def replace_wildcards(self, cards, replacement):
        """替换牌中的逢人配"""
        return [f'替换{replacement}' if isinstance(card, str) and '红桃' in card and self.level_card in card else card
                for card in cards]

    def is_valid_play(self, cards):
        """判断出牌是否合法（支持逢人配）"""
        if not cards:
            return False
        if not self.has_wildcard(cards):
            return self._is_valid_play_without_wildcard(cards)

        possible_replacements = range(2, 15)  # 逢人配可变为 2~A

        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            if self._is_valid_play_without_wildcard(new_cards):
                return True
        return False

    def _is_valid_play_without_wildcard(self, cards):
        """原始牌型判断逻辑"""
        length = len(cards)
        if length == 1:
            return True
        if length == 2:
            return self.is_pair(cards)
        if length == 3:
            return self.is_triple(cards)
        if length == 4:
            return self.is_king_bomb(cards) or self.is_bomb(cards)
        if length == 5:
            return self.is_straight(cards) or self.is_flush_straight(cards) or self.is_three_with_two(cards) or self.is_bomb(cards)
        if length == 6:
            return self.is_triple_pair(cards) or self.is_triple_consecutive(cards) or self.is_bomb(cards)
        if 6 < length <= 8:
            return self.is_bomb(cards)
        return False

    def is_king_bomb(self, cards):
        """四大天王（天王炸）"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def is_pair(self, cards):
        """对子（考虑逢人配）"""
        if len(cards) != 2:
            return False
        if not self.has_wildcard(cards):
            return self.get_rank(cards[0]) == self.get_rank(cards[1])
        possible_replacements = range(2, 15)  # 逢人配可变 2~A
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            if self.get_rank(new_cards[0]) == self.get_rank(new_cards[1]):
                return True
        return False

    def is_triple(self, cards):
        """三同张（考虑逢人配）"""
        if len(cards) != 3:
            return False
        if not self.has_wildcard(cards):
            return len(set(self.get_rank(card) for card in cards)) == 1
        possible_replacements = range(2, 15)
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            if len(set(self.get_rank(card) for card in new_cards)) == 1:
                return True
        return False

    def is_three_with_two(self, cards):
        """三带二（考虑逢人配）"""
        if len(cards) != 5:
            return False
        if not self.has_wildcard(cards):
            counts = Counter(self.get_rank(card) for card in cards)
            return 3 in counts.values() and 2 in counts.values()
        possible_replacements = range(2, 15)
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            counts = Counter(self.get_rank(card) for card in new_cards)
            if 3 in counts.values() and 2 in counts.values():
                return True
        return False

    def is_straight(self, cards):
        """顺子（考虑 A=1 和 A=14，逢人配也必须合理变换）"""
        if len(cards) != 5:
            return False
        if not self.has_wildcard(cards):
            ranks = sorted(self.get_rank(card, as_one=False) for card in cards)
            ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in cards)
            return self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one)

        possible_replacements = range(2, 15)  # 逢人配可变 2~A
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            ranks = sorted(self.get_rank(card, as_one=False) for card in new_cards)
            ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in new_cards)
            if self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one):
                return True
        return False

    def is_bomb(self, cards):
        """炸弹（考虑逢人配）"""
        if len(cards) < 4:
            return False
        if not self.has_wildcard(cards):
            return len(set(self.get_rank(card) for card in cards)) == 1
        possible_replacements = range(2, 15)
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            if len(set(self.get_rank(card) for card in new_cards)) == 1:
                return True
        return False

    def is_triple_consecutive(self, cards):
        """钢板（考虑逢人配）"""
        if len(cards) != 6:
            return False
        if not self.has_wildcard(cards):
            ranks = sorted(self.get_rank(card) for card in cards)
            counter = Counter(ranks)
            triples = [rank for rank, count in counter.items() if count == 3]
            return len(triples) == 2 and self._is_consecutive(triples)
        possible_replacements = range(2, 15)
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            ranks = sorted(self.get_rank(card) for card in new_cards)
            counter = Counter(ranks)
            triples = [rank for rank, count in counter.items() if count == 3]
            if len(triples) == 2 and self._is_consecutive(triples):
                return True
        return False

    def is_triple_pair(self, cards):
        """连对（木板，考虑逢人配）"""
        if len(cards) != 6:
            return False
        if not self.has_wildcard(cards):
            ranks = sorted(self.get_rank(card) for card in cards)
            counter = Counter(ranks)
            pairs = [rank for rank, count in counter.items() if count == 2]
            return len(pairs) == 3 and self._is_consecutive(pairs)
        possible_replacements = range(2, 15)
        for replacement in possible_replacements:
            new_cards = self.replace_wildcards(cards, replacement)
            ranks = sorted(self.get_rank(card) for card in new_cards)
            counter = Counter(ranks)
            pairs = [rank for rank, count in counter.items() if count == 2]
            if len(pairs) == 3 and self._is_consecutive(pairs):
                return True
        return False

    def is_flush_straight(self, cards):
        """同花顺（火箭），逢人配可变点数和花色"""
        if len(cards) != 5:
            return False

        # 提取已有的花色（去掉逢人配）
        suits = {card[:2] for card in cards if isinstance(card, str) and '红桃' not in card}

        # 可能的最终花色（如果有逢人配，它可以变成任何花色）
        if len(suits) == 1:
            possible_suits = suits
        else:
            possible_suits = ['黑桃', '红桃', '梅花', '方块']

        # 可能的数值替换
        possible_replacements = range(2, 15)  # 逢人配可变为 3~A

        # 遍历所有可能的花色和点数组合
        for suit in possible_suits:
            for replacement in possible_replacements:
                new_cards = [
                    f'{suit}{replacement}' if '红桃' in card and self.level_card in card else card
                    for card in cards
                ]
                suits = {card[:2] for card in new_cards if isinstance(card, str)}
                ranks = sorted(self.get_rank(card) for card in new_cards)

                # A=1 或 A=14 的情况
                ranks_as_one = sorted(self.get_rank(card, as_one=True) for card in new_cards)

                if len(suits) == 1 and (self._is_consecutive(ranks) or self._is_consecutive(ranks_as_one)):
                    return True
        return False

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

    def get_rank(self, card, as_one=False, replacement=None):
        """获取牌点数，逢人配可以替换，A 可以视为 1 或 14"""
        if card in ['小王', '大王']:
            return CARD_RANKS[card]

        rank = card[2:] if len(card) > 2 else card[2]

        # **如果是逢人配**
        if '红桃' in card and self.level_card in card:
            return 15 if replacement is None else replacement  # 单独打出时是 15，替换时按 replacement

        # **A 的特殊处理**
        if rank == 'A':
            return 1 if as_one else 14  # A 可以视为 1 或 14

        return CARD_RANKS.get(rank, 0)

    def _is_consecutive(self, ranks):
        """判断是否为连续数字序列"""
        return all(ranks[i] == ranks[i - 1] + 1 for i in range(1, len(ranks)))

    def can_beat(self, previous_play, current_play):
        """判断当前出牌是否能压过上家（考虑逢人配）"""
        if not self.is_valid_play(current_play):
            return False
        if not previous_play:
            return True  # 无上家出牌，直接出

        prev_type = self.get_play_type(previous_play)
        curr_type = self.get_play_type(current_play)

        # 牌力排序（炸弹优先级）
        bomb_order = ['天王炸', '8炸', '7炸', '6炸', '同花顺', '5炸', '4炸']

        # **炸弹能压制非炸弹**
        if curr_type in bomb_order and prev_type not in bomb_order:
            return True
        if prev_type in bomb_order and curr_type in bomb_order:
            return bomb_order.index(curr_type) < bomb_order.index(prev_type)

        # **如果牌型不同，不能比较**
        if prev_type != curr_type:
            return False

        # **逢人配处理**
        if self.has_wildcard(current_play):
            possible_replacements = range(2, 15)
            for replacement in possible_replacements:
                new_play = self.replace_wildcards(current_play, replacement)
                if self.get_play_value(new_play) > self.get_play_value(previous_play):
                    return True
            return False

        return self.get_play_value(current_play) > self.get_play_value(previous_play)

    def get_play_value(self, cards):
        """获取牌点数，若有逢人配，计算最优替换"""
        ranks = [self.get_rank(card) for card in cards]
        return max(ranks)


if __name__ == "__main__":
    current_round = 2
    #rules = Rules(level_card=str(current_round))
    rules = Rules(level_card='2')

    # ✅ 合法情况

    print(rules.is_valid_play(['红桃10','红桃J','红桃Q','红桃2','黑桃A']))






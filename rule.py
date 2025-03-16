from collections import Counter

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

class Rules:
    def __init__(self, level_card=None):
        self.level_card = level_card  # 级牌

    def is_valid_play(self, cards):
        """判断出牌是否合法"""
        if not cards:
            return False
        length = len(cards)

        if length == 1:
            return True  # 单张
        if length == 2:
            return self.is_pair(cards) or self.is_rocket(cards)  # 对子 or 王炸
        if length == 3:
            return self.is_triple(cards)  # 三同张
        if length == 4:
            return self.is_king_bomb(cards) or self.is_bomb(cards)  # 天王炸 or 炸弹
        if length == 5:
            return self.is_straight(cards) or self.is_three_with_two(cards)  # 顺子 or 三带二
        if length == 6:
            return self.is_triple_pair(cards) or self.is_triple_consecutive(cards)  # 连对（木板） or 三同连张（钢板）
        return self.is_bomb(cards)  # 5 张及以上的炸弹

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
        """连对（木板）"""
        if len(cards) != 6:
            return False
        counts = Counter(self.get_rank(card) for card in cards)
        pairs = [rank for rank, count in counts.items() if count == 2]
        return len(pairs) == 3 and self._is_consecutive(pairs)

    def is_triple_consecutive(self, cards):
        """三同连张（钢板，如 555666）"""
        if len(cards) != 6:
            return False
        counts = Counter(self.get_rank(card) for card in cards)
        pairs = [rank for rank, count in counts.items() if count == 2]
        # 需要恰好是 3 个对子，且必须连续
        return len(pairs) == 3 and self._is_consecutive(pairs)



    def is_straight(self, cards):
        """顺子（必须 5 张，A 可作为 1）"""
        if len(cards) != 5:
            return False
        ranks = sorted(self.get_rank(card, as_one=False) for card in cards)

        if ranks[-1] == 14:  # A 作为 14
            alt_ranks = ranks[:-1] + [1]
            alt_ranks.sort()
            if self._is_consecutive(alt_ranks):
                return True

        return self._is_consecutive(ranks)

    def is_bomb(self, cards):
        """炸弹（5 张及以上的相同牌 or 4 张相同牌）"""
        if len(cards) < 4:
            return False
        ranks = [self.get_rank(card) for card in cards]
        return len(set(ranks)) == 1

    def is_rocket(self, cards):
        """王炸"""
        return set(cards) == {'小王', '大王'}

    def is_king_bomb(self, cards):
        """四大天王（天王炸）"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def get_rank(self, card, as_one=False):
        """获取牌点数，支持 A 作为 1"""
        if card in ['小王', '大王']:
            return CARD_RANKS[card]
        rank = card[2:] if len(card) > 2 else card[2]
        if as_one and rank == 'A':
            return 1
        if self.level_card and self.level_card in rank:
            return CARD_RANKS['A'] + 1
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

        # 炸弹规则（能压制非炸弹）
        bomb_order = ['天王炸', '大炸弹', '王炸', '炸弹']
        if curr_type in bomb_order and prev_type not in bomb_order:
            return True
        if prev_type in bomb_order and curr_type in bomb_order:
            return bomb_order.index(curr_type) < bomb_order.index(prev_type)

        # 牌型必须相同才能比较
        if prev_type != curr_type:
            return False

        return self.get_play_value(current_play) > self.get_play_value(previous_play)

    def get_play_type(self, cards):
        """获取牌型"""
        if self.is_king_bomb(cards):
            return '天王炸'
        if self.is_rocket(cards):
            return '王炸'
        if self.is_bomb(cards):
            return '炸弹' if len(cards) == 4 else '大炸弹'
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




if __name__ == "__main__":
    current_round = 2
    #rules = Rules(level_card=str(current_round))
    rules = Rules(level_card=None)

    # **测试各种牌型**
    print(rules.is_valid_play(['黑桃10' ,'黑桃10' ,'黑桃9', '梅花9' ,'黑桃8', '红桃8']))  # ✅ True（天王炸）
    print(rules.is_valid_play(['黑桃A', '红桃2', '方块3', '梅花4', '黑桃5']))  # ✅ True（顺子）
    print(rules.is_valid_play(['黑桃10', '黑桃J', '黑桃Q', '黑桃K', '黑桃A']))  # ✅ True（顺子）


    # 测试出牌规则
    print(rules.can_beat(['黑桃3', '红桃3'], ['黑桃2', '红桃2']))  # 输出: False (2 是级牌，大于 3)
    print(rules.can_beat(['黑桃A', '红桃A'], ['黑桃K', '红桃K']))
    print(rules.can_beat(['黑桃3', '红桃3'], ['黑桃2', '红桃2']))  # False
    print(rules.can_beat(['黑桃A', '红桃A'], ['黑桃K', '红桃K']))  # True
    print(rules.can_beat(['黑桃5', '红桃5', '方块5', '梅花5', '梅花5'], ['小王', '大王']))  # True（王炸压制炸弹）
    print(rules.can_beat(['小王', '小王', '大王', '大王'], ['黑桃5', '红桃5', '方块5', '梅花5', '黑桃5']))  # False（天王炸最大）
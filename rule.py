"""
rule.py
掼蛋核心规则判定模块（纯Python）。
主要功能：判断出牌合法性、比较牌型大小、特殊结构（如三带二、钢板等）判定、辅助判定函数。
支持点数、花色、级牌等多元规则。
"""

from collections import Counter

# 定义牌的点数
CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '小王': 16, '大王': 17
}

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']



class Rules:
    def __init__(self, level_card=None):
        self.level_card = level_card  # 级牌
        self.CARD_RANKS = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
            '小王': 16, '大王': 17
        }
        self.RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        self.PLAY_MAPPING = {
            "天王炸":100,
            "同花顺":101,
            "4炸":102,
            "5炸":103,
            "6炸":104,
            "7炸":105,
            "8炸":106,
            "钢板":107,
            "木板":108,
            "三带二":109,
            "三同张":110,
            "顺子":111,
            "对子":112,
            "单牌":113,
            "非法牌型":1014
        }
        self.comment = None
        
    def _is_wildcard(self, card):
        """判断一张牌是否为赖子（红桃级牌）"""
        # 确保card是字符串类型
        if not isinstance(card, str):
            return False
            
        # 检查是否为红桃级牌
        return card.startswith('红桃') and self.get_rank(card, ignore_level=True) == self.level_card

    def is_valid_play(self, cards):
        """判断出牌是否合法，支持红桃级牌作为赖子，返回(是否合法, 牌型)"""
        if not cards:
            return (False, "空牌")
            
        length = len(cards)
        # 获取牌型名称
        play_type = self.get_play_type(cards)
        
        # 统计赖子数量
        wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        
        # 1. 单张
        if length == 1:
            return (True, play_type)
        
        # 2. 对子
        if length == 2:
            result = self.is_pair(cards, wildcards)
            return (result, play_type)
        
        # 3. 三同张
        if length == 3:
            result = self.is_triple(cards, wildcards)
            return (result, play_type)
        
        # 4. 炸弹（4张及以上）
        if length >= 4:
            if self.is_king_bomb(cards):
                return (True, play_type)
            if self.is_bomb(cards, wildcards):
                return (True, play_type)
        
        # 5. 顺子（5张及以上）
        if length >= 5:
            if self.is_flush_straight(cards, wildcards):
                return (True, play_type)
            if self.is_straight(cards, wildcards):
                return (True, play_type)
        
        # 6. 三带二
        if length == 5:
            if self.is_three_with_two(cards, wildcards):
                return (True, play_type)

        # 7. 钢板（2组及以上） 333444 钢板只允许6张
        if length >= 6 and length % 3 == 0:
            if self.is_triple_consecutive(cards, wildcards):
                return (True, play_type)
                
        # 8. 连对（3对及以上） 33445566  333444555666
        if length >= 6 and length % 2 == 0:
            if self.is_triple_pair(cards, wildcards):
                return (True, play_type)
                
        # 9. 明确排除"四带一"牌型（5张牌）
        if length == 5:
            # 统计所有牌的点数（包括赖子）
            all_points = [self.get_rank(card) for card in cards]
            point_counts = Counter(all_points)
            
            # 检查是否存在4张相同点数和1张不同点数
            if len(point_counts) == 2:
                counts = list(point_counts.values())
                if (4 in counts and 1 in counts) or (1 in counts and 4 in counts):
                    return (False, play_type)
                    
            # 考虑赖子情况：3张相同点数+1张赖子+1张其他点数
            if wildcard_count == 1:
                non_wild_points = [self.get_rank(card) for card in non_wildcards]
                non_wild_counts = Counter(non_wild_points)
                
                # 检查是否有3张相同点数和1张其他点数
                if len(non_wild_counts) == 2:
                    counts = list(non_wild_counts.values())
                    if 3 in counts and 1 in counts:
                        return (False, play_type)
        
        # 10. 特殊牌型：天王炸
        if self.is_king_bomb(cards):
            return (True, play_type)
        
        return (False, play_type)

    def is_pair(self, cards, wildcards=None):
        """对子，支持赖子"""
        if len(cards) != 2:
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 情况1: 两张都是赖子
        if wildcard_count == 2:
            return True
            
        # 情况2: 一张赖子 + 一张普通牌
        if wildcard_count == 1:
            return True
            
        # 情况3: 没有赖子，两张牌点数相同
        return self.get_rank(cards[0]) == self.get_rank(cards[1])

    def is_triple(self, cards, wildcards=None):
        """三同张（三不带），支持赖子"""
        if len(cards) != 3:
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 情况1: 三张都是赖子
        if wildcard_count == 3:
            return True
            
        # 情况2: 两张赖子 + 一张普通牌
        if wildcard_count == 2:
            return True
            
        # 情况3: 一张赖子 + 两张普通牌
        if wildcard_count == 1:
            non_wildcards = [card for card in cards if not self._is_wildcard(card)]
            return self.get_rank(non_wildcards[0],ignore_level=True) == self.get_rank(non_wildcards[1],ignore_level=True)
            
        # 情况4: 没有赖子，三张牌点数相同
        return len(set(self.get_rank(card) for card in cards)) == 1

    def is_three_with_two(self, cards, wildcards=None):
        """三带二，支持赖子，但一张赖子只能被使用一次（要么在三张部分，要么在对子部分）"""
        if len(cards) != 5:
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 统计非赖子牌的点数分布
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        counts = Counter(self.get_rank(card) for card in non_wildcards)
        
        # 检查是否可能组成三带二
        for trio_point in counts:
            # 计算三张部分需要的赖子数
            trio_need = max(0, 3 - counts[trio_point])
            if trio_need > wildcard_count:
                continue  # 这个点数无法组成三张
            
            # 计算剩余赖子数
            remaining_wildcards = wildcard_count - trio_need
            
            # 检查对子部分
            for pair_point in counts:
                if trio_point == pair_point:
                    # 相同点数：需要检查该点数在组成三张后剩余的牌
                    remaining_cards = counts[pair_point] - min(counts[pair_point], 3)
                    pair_need = max(0, 2 - remaining_cards)
                else:
                    # 不同点数：直接检查该点数的牌数
                    pair_need = max(0, 2 - counts[pair_point])
                
                # 关键修改：确保赖子不被重复使用
                if pair_need <= remaining_wildcards:
                    return True
        
        # 检查完全由赖子组成的情况
        if wildcard_count >= 5:
            return True
            
        # 尝试赖子用于对子部分的情况
        for pair_point in counts:
            # 计算对子部分需要的赖子数
            pair_need = max(0, 2 - counts[pair_point])
            if pair_need > wildcard_count:
                continue
                
            # 计算剩余赖子数
            remaining_wildcards = wildcard_count - pair_need
            
            # 检查三张部分
            for trio_point in counts:
                if trio_point == pair_point:
                    # 相同点数：需要检查该点数在组成对子后剩余的牌
                    remaining_cards = counts[trio_point] - min(counts[trio_point], 2)
                    trio_need = max(0, 3 - remaining_cards)
                else:
                    # 不同点数：直接检查该点数的牌数
                    trio_need = max(0, 3 - counts[trio_point])
                
                if trio_need <= remaining_wildcards:
                    return True
                    
        return False
    
    def is_triple_pair(self, cards, wildcards=None):
        """连对（木板），如 556677，支持赖子"""
        if len(cards) < 6 or len(cards) % 2 != 0:
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 获取所有非赖子牌的点数
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        non_wild_ranks = [self.get_rank(card,ignore_level=True) for card in non_wildcards]
        counts = Counter(non_wild_ranks)
        
        # 检查每个点数是否都是对子（允许赖子补足）
        for rank, count in counts.items():
            # 如果某个点数出现超过2次，则不符合连对要求
            if count > 2:
                return False
                
        # 计算每个点数需要补足的对子数
        missing_pairs = 0
        for rank, count in counts.items():
            if count < 2:
                missing_pairs += (2 - count)
        
        # 如果缺失的对子数超过赖子数量，则无法组成连对
        if missing_pairs > wildcard_count:
            return False
            
        # 获取所有非赖子牌的点数（去重并排序）
        unique_ranks = sorted(set(non_wild_ranks))
        if not unique_ranks:
            # 全是赖子，可以组成任意连对
            return True
            
        # 检查点数是否连续（允许赖子填补空缺）
        min_rank = min(unique_ranks)
        max_rank = max(unique_ranks)
        
        # 检查实际点数序列是否连续
        if not self._is_consecutive(unique_ranks):
            # 点数不连续，需要赖子填补空缺
            gap_count = 0
            for i in range(1, len(unique_ranks)):
                gap = unique_ranks[i] - unique_ranks[i-1] - 1
                if gap > 0:
                    gap_count += gap
                    
            # 如果空缺数超过赖子数量，则无法组成连对
            if gap_count > wildcard_count - missing_pairs:
                return False
                
        return True
    
    def is_triple_consecutive(self, cards, wildcards=None):
        """三同连张（钢板），如 555666，支持赖子"""
        # 钢板必须由2组及以上连续三张组成
        if len(cards) < 6 or len(cards) % 3 != 0:
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 获取所有非赖子牌的点数
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        non_wild_ranks = [self.get_rank(card,ignore_level=True) for card in non_wildcards]
        counts = Counter(non_wild_ranks)
        
        # 计算每个点数需要补足的三张数
        missing_trios = 0
        for rank, count in counts.items():
            if count < 3:
                missing_trios += (3 - count)
        
        # 如果缺失的三张数超过赖子数量，则无法组成钢板
        if missing_trios > wildcard_count:
            return False
            
        # 获取所有非赖子牌的点数（去重并排序）
        unique_ranks = sorted(set(non_wild_ranks))
        if not unique_ranks:
            # 全是赖子，可以组成任意钢板
            return True
            
        # 确保所有非赖子牌的点数都在连续序列中
        min_rank = min(unique_ranks)
        max_rank = max(unique_ranks)
        
        # 检查点数是否连续（允许赖子填补空缺）
        if not self._is_consecutive(unique_ranks):
            # 点数不连续，需要赖子填补空缺
            gap_count = 0
            for i in range(1, len(unique_ranks)):
                gap = unique_ranks[i] - unique_ranks[i-1] - 1
                if gap > 0:
                    gap_count += gap
                    
            # 如果空缺数超过赖子数量，则无法组成钢板
            if gap_count > wildcard_count - missing_trios:
                return False
                
        # 确保没有多余的点数（即所有牌都用于组成连续三同张）
        total_cards_needed = (max_rank - min_rank + 1) * 3
        if len(cards) != total_cards_needed:
            return False
            
        return True

    def is_straight(self, cards, wildcards=None):
        """顺子（5张及以上），支持赖子，但小王和大王不能构成顺子"""
        if len(cards) != 5:
            return False

        # 检查是否包含大王或小王
        if any(card in ['大王', '小王'] for card in cards):
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 获取所有非赖子牌的点数
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        ranks = [self.get_rank(card, as_one=False, ignore_level=True) for card in non_wildcards]
        
        # 检查非赖子牌中是否有重复点数（重复点数不能用赖子替换）
        if len(ranks) != len(set(ranks)):
            return False
            
        points = sorted(set(ranks))
        
        # 计算顺子的最小值和最大值
        min_point = min(points)
        max_point = max(points)
        
        # 计算点数序列中的空缺数
        gaps = 0
        gaps_list = []
        for i in range(1, len(points)):
            gap = points[i] - points[i-1] - 1
            if gap > 0:
                gaps += gap
                gaps_list.append(points[i]+1)
        
        #中间不缺点数
        if gaps == 0:
            if wildcard_count <= min_point-2 :
                return True
            if wildcard_count <= 14-max_point :
                return True
            
        if min_point<=2 and max_point>=14 and gaps<wildcard_count:
            return False
            
        # 如果空缺数不超过赖子数量，则可以组成顺子
        return gaps <= wildcard_count

    def is_flush_straight(self, cards, wildcards=None):
        """同花顺（火箭），支持赖子"""
        if len(cards) != 5:
            return False

        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 获取所有非赖子牌的花色
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        
        # 如果存在非赖子牌，检查它们的花色是否一致
        if non_wildcards:
            # 正确获取花色：取前两个字符（如"梅花"、"红桃"等）
            base_suit = non_wildcards[0][:2]
            for card in non_wildcards[1:]:
                # 同样取前两个字符作为花色
                if card[:2] != base_suit:
                    return False
            
        # 检查点数是否连续（使用is_straight方法）
        if not self.is_straight(cards, wildcards):
            return False
            
        return True

    def is_bomb(self, cards, wildcards=None):
        """炸弹（4张及以上相同点数），支持赖子"""
        if len(cards) < 4:
            return False
            
        if wildcards is None:
            wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        
        # 获取所有非赖子牌的点数
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        if not non_wildcards:
            return True  # 全是赖子，可以组成炸弹
            
        # 检查非赖子牌的点数是否相同
        base_point = self.get_rank(non_wildcards[0],ignore_level=True)
        for card in non_wildcards[1:]:
            if self.get_rank(card) != base_point:
                return False
                
        return True

    def is_king_bomb(self, cards):
        """四大天王（天王炸）"""
        return sorted(cards) == ['大王', '大王', '小王', '小王']

    def get_rank(self, card, as_one=False, ignore_level=False, wildcard_value=None):
        """
        获取牌的点数，支持 A=1
        wildcard_value: 当级牌作为赖子使用时，指定其充当的点数
        """
        # 确保card是字符串类型
        if not isinstance(card, str):
            return card  # 如果已经是点数，直接返回
            
        if card in ['小王', '大王']:
            return CARD_RANKS[card]

        # 解析点数
        if card.startswith('红桃') or card.startswith('黑桃') or card.startswith('梅花') or card.startswith('方块'):
            rank = card[2:]  # 格式为"花色点数"
        else:
            rank = card  # 已经是点数字符串
            
        # 当级牌作为赖子使用时，返回其充当的点数
        if wildcard_value is not None and self._is_wildcard(card):
            return wildcard_value
            
        # **只检查当前局的级牌（除非ignore_level=True）**
        if not ignore_level and self.level_card and rank == self.RANKS[self.level_card - 2]:
            return self.CARD_RANKS['A'] + 1  # 级牌比 A 还大

        if as_one and rank == 'A':
            return 1  # A 作为 1

        return CARD_RANKS.get(rank, 0)

    def _is_consecutive(self, ranks):
        """判断是否为连续数字序列"""
        return all(ranks[i] == ranks[i - 1] + 1 for i in range(1, len(ranks)))

    def can_beat(self, previous_play, current_play):
        """
        判断当前出牌是否能压过上家，考虑级牌和赖子
        返回: (是否能压过, 当前牌型, 当前最大点数)
        """
        # 检查当前出牌是否合法
        current_valid, current_type = self.is_valid_play(current_play)
        if not current_valid:
            return (False, current_type, 0)
            
        # 获取当前牌型的最大点数
        current_max = self.get_play_value(current_play)
        
        if not previous_play:
            return (True, current_type, current_max)  # 没人出牌，可以随便出

        # 获取上一手牌的牌型
        prev_valid, prev_type = self.is_valid_play(previous_play)
        if not prev_valid:
            return (True, current_type, current_max)  # 上一手不合法，当前牌可以压过

        # 获取上一手牌的最大点数
        prev_max = self.get_play_value(previous_play)

        # 定义特殊炸弹类型
        special_bombs = ['天王炸', '同花顺']
        # 普通炸弹类型（按牌数从大到小排序）
        common_bombs = ['8炸', '7炸', '6炸', '5炸', '4炸']

        # **炸弹能压制非炸弹**
        if (current_type in special_bombs or current_type in common_bombs) and (prev_type not in special_bombs and prev_type not in common_bombs):
            return (True, current_type, current_max)

        # **炸弹之间比较**
        if (prev_type in special_bombs or prev_type in common_bombs) and (current_type in special_bombs or current_type in common_bombs):
            # 先处理特殊炸弹
            if prev_type in special_bombs and current_type in special_bombs:
                # 两个都是特殊炸弹，按照固定顺序：天王炸 > 同花顺
                special_order = ['天王炸', '同花顺']
                result = special_order.index(current_type) < special_order.index(prev_type)
                return (result, current_type, current_max)
            elif prev_type in special_bombs and current_type in common_bombs:
                # 上一手是特殊炸弹，当前是普通炸弹，不能压过
                return (False, current_type, current_max)
            elif current_type in special_bombs and prev_type in common_bombs:
                # 当前是特殊炸弹，上一手是普通炸弹，可以压过
                return (True, current_type, current_max)
            else:
                # 两个都是普通炸弹
                # 比较牌数：牌数多的炸弹大
                prev_size = len(previous_play)
                curr_size = len(current_play)
                if curr_size > prev_size:
                    return (True, current_type, current_max)
                elif curr_size < prev_size:
                    return (False, current_type, current_max)
                else:
                    # 牌数相同，比较点数
                    result = current_max > prev_max
                    return (result, current_type, current_max)

        # **牌型必须相同才能比较**
        if prev_type != current_type:
            return (False, current_type, current_max)

        # **特殊牌型比较规则**
        if prev_type == '三带二' and current_type == '三带二':
            # 三带二比较三张部分的大小
            prev_trio_value = self._get_trio_value(previous_play)
            curr_trio_value = self._get_trio_value(current_play)
            result = curr_trio_value > prev_trio_value
            return (result, current_type, current_max)
            
        if prev_type == '钢板' and current_type == '钢板':
            # 钢板必须长度相同才能比较
            if len(previous_play) != len(current_play):
                return (False, current_type, current_max)
            # 钢板比较最大牌的大小（因为钢板是连续三同张，比较最后一组牌的点数）
            result = current_max > prev_max
            return (result, current_type, current_max)
            
        if prev_type == '木板' and current_type == '木板':
            # 木板必须长度相同才能比较
            if len(previous_play) != len(current_play):
                return (False, current_type, current_max)
            # 木板比较最大牌的大小（因为木板是连续对子，比较最后一对牌的点数）
            result = current_max > prev_max
            return (result, current_type, current_max)

        if prev_type == '同花顺' and current_type == '同花顺':
            # 顺子必须长度相同才能比较
            if len(previous_play) != len(current_play):
                return (False, current_type, current_max)
            # 顺子比较最大牌的大小
            result = current_max > prev_max
            return (result, current_type, current_max)
        
        if prev_type == '顺子' and current_type == '顺子':
            # 顺子必须长度相同才能比较
            if len(previous_play) != len(current_play):
                return (False, current_type, current_max)
            # 顺子比较最大牌的大小
            result = current_max > prev_max
            return (result, current_type, current_max)

        # **普通牌型比较点数，考虑级牌和赖子**
        # 如果当前牌型包含赖子（红桃级牌），可以当作任意牌使用
        has_wildcard = any(self._is_wildcard(card) for card in current_play)
        
        # 如果当前牌型包含赖子，且点数相等，可以压过
        if has_wildcard and current_max >= prev_max:
            return (True, current_type, current_max)
            
        # 普通比较
        result = current_max > prev_max
        return (result, current_type, current_max)

    def _get_trio_value(self, cards):
        """获取三带二牌型中三张部分的值"""
        ranks = [self.get_rank(card) for card in cards]
        count = Counter(ranks)
        # 找出出现次数最多的点数（三张部分）
        trio_value = max(count.items(), key=lambda x: x[1])[0]
        return trio_value

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
        """获取牌点数（最大值），考虑赖子充当的点数"""
        type = self.get_play_type(cards)
        
        # 获取赖子数量
        wildcards = [card for card in cards if self._is_wildcard(card)]
        wildcard_count = len(wildcards)
        non_wildcards = [card for card in cards if not self._is_wildcard(card)]
        
        if type == '天王炸':
            return 1000  # 天王炸是最大的牌型
            
        elif type == '同花顺':
            # 同花顺的最大点数就是顺子的最大点数
            return self._get_straight_max_value(non_wildcards, wildcard_count)
            
        elif type in ['4炸', '5炸', '6炸', '7炸', '8炸']:
            # 炸弹点数由非赖子牌决定（如果有非赖子牌）
            if non_wildcards:
                return self.get_rank(non_wildcards[0],ignore_level=True)
            # 全是赖子时，返回级牌点数
            return self.get_rank(wildcards[0])
            
        elif type == '钢板':
            # 钢板的最大点数就是连续三同张的最大点数
            # 获取所有非赖子牌的点数（去重并排序）
            points = sorted(set(self.get_rank(card,ignore_level=True) for card in non_wildcards))
            if not points:
                # 全是赖子，返回级牌点数
                return self.get_rank(wildcards[0])
            # 最大点数 = 最小点数 + (组数 - 1)
            min_point = min(points)
            group_count = len(cards) // 3
            return min_point + group_count - 1
            
        elif type == '木板':
            # 木板的最大点数就是连续对子的最大点数
            # 获取所有非赖子牌的点数（去重并排序）
            points = sorted(set(self.get_rank(card,ignore_level=True) for card in non_wildcards))
            if not points:
                # 全是赖子，返回级牌点数
                return self.get_rank(wildcards[0])
            # 最大点数 = 最小点数 + (对子数 - 1)
            min_point = min(points)
            pair_count = len(cards) // 2
            return min_point + pair_count - 1
            
        elif type == '三带二':
            # 三带二比较三张部分的大小
            return self._get_trio_value(cards)
            
        elif type == '三同张':
            if non_wildcards:
                return self.get_rank(non_wildcards[0],ignore_level=True)
            return self.get_rank(wildcards[0])
            
        elif type == '顺子':
            return self._get_straight_max_value(non_wildcards, wildcard_count)
            
        elif type == '对子':
            if non_wildcards:
                return self.get_rank(non_wildcards[0],ignore_level=True)
            return self.get_rank(wildcards[0])
            
        elif type == '单牌':
            return self.get_rank(cards[0])
            
        else:
            return 0  # 非法牌型返回0
        
    def _get_straight_max_value(self, non_wildcards, wildcard_count):
        """计算顺子的最大点数（包括同花顺），考虑赖子可以充当更高点数"""
        # 获取非赖子牌的点数（去重并排序）
        points = sorted(set(self.get_rank(card,ignore_level=True) for card in non_wildcards))
        if not points:
            # 全是赖子，最大点数为14（A）
            return 14
            
        # 计算顺子的最大可能点数
        max_value = 0
        # 尝试从14（A）开始向下寻找可能的顺子
        for top in range(14, 4, -1):
            missing = 0
            # 检查从top-4到top的连续5个点数
            for p in range(top-4, top+1):
                if p not in points:
                    missing += 1
            # 如果缺失的点数不超过赖子数量，则这个顺子成立
            if missing <= wildcard_count:
                max_value = top
                break
                
        # 考虑赖子可以充当更高点数的情况
        # 获取当前最大点数
        current_max = max(points) if points else 0
        # 计算赖子可以扩展的点数
        if wildcard_count > 0 and current_max < 14:
            # 赖子可以充当比当前最大点数更高的点数
            # 最多可以扩展到14（A）
            possible_extension = min(14 - current_max, wildcard_count)
            # 新的最大点数 = 当前最大点数 + possible_extension
            new_max = current_max + possible_extension
            # 如果新的最大点数大于之前计算的max_value，则更新
            if new_max > max_value:
                max_value = new_max
                
        return max_value
    
    def get_play_value_min(self, cards):
        """获取牌点数（最小值）"""
        ranks = [self.get_rank(card) for card in cards]
        return min(ranks)

    def cards_to_mod_id(self, cards):
        """
        将输入的牌ID转换为拼接的取余数字
        规则：
          1. 创建4x16的二维数组（黑桃、红桃、方块、梅花）
          2. 大王放在红桃16位置，小王放在黑桃16位置
          3. 其他牌按花色和点数放入对应位置
          4. 每行转换为十六进制数，对997取余
          5. 将四个取余结果拼接为字符串
        """
        # 初始化4x16的二维数组，全0表示无牌
        suits = ["黑桃", "红桃", "方块", "梅花"]
        card_map = [[0] * 16 for _ in range(4)]
        
        for card in cards:
            if card == "大王":
                # 大王放在红桃16位置（红桃行索引1，位置15）
                card_map[1][15] = 1
            elif card == "小王":
                # 小王放在黑桃16位置（黑桃行索引0，位置15）
                card_map[0][15] = 1
            else:
                # 解析花色和点数
                suit = card[:2]
                rank_str = card[2:]
                
                # 获取花色索引
                suit_idx = suits.index(suit)
                
                # 获取点数对应的位置（点数2对应位置0，点数A对应位置12）
                rank_val = self.CARD_RANKS.get(rank_str, 0)
                if rank_val >= 2 and rank_val <= 14:  # 普通牌点数范围2-A
                    pos = rank_val - 2
                    card_map[suit_idx][pos] = 1
        
        # 将每行转换为十六进制数并取余
        mod_results = []
        for row in card_map:
            # 将每行16个二进制位转换为整数
            bin_str = ''.join(str(bit) for bit in row)
            int_value = int(bin_str, 2)
            # 对997取余
            mod_value = int_value % 997
            mod_results.append(str(mod_value))
        
        # 拼接四个取余结果
        mod_str = ''.join(mod_results)
        # 将字符串转换为整数
        mod_int = int(mod_str)
        # 对1024取余
        result = mod_int % 1023 + 1
        return result

if __name__ == "__main__":
    rules = Rules(level_card=10)
    #print(rules.is_valid_play(['黑桃8', '红桃8', '方块7', '梅花9', '黑桃7', '黑桃9']))
    #print(rules.can_beat(['黑桃8', '红桃8', '方块7', '梅花9', '黑桃7', '黑桃9'],['黑桃8', '红桃8', '方块10', '梅花9', '黑桃10', '黑桃9']))
    #print(rules.can_beat(['黑桃8', '红桃9', '方块10', '梅花J', '黑桃Q', '黑桃K'],['黑桃9', '红桃10', '方块J', '梅花Q', '黑桃K', '黑桃A']))
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花8', '黑桃8', '黑桃8'],['黑桃9', '红桃9', '方块9', '梅花9', '黑桃9', '黑桃9']))
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花8', '黑桃8', '黑桃8'],['黑桃9', '红桃9', '方块9', '梅花9', '黑桃9', '红桃10']))
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花9', '黑桃9'],['黑桃9', '红桃9', '方块10', '梅花9', '黑桃10']))
    #print(rules.can_beat(['黑桃8'],['黑桃9']))
    #print(rules.can_beat(['黑桃8', '红桃8'],['方块10', '黑桃10']))
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花9', '黑桃9', '黑桃9'],['黑桃10', '红桃10', '方块10', '梅花9', '黑桃9', '黑桃9']))
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花9', '黑桃9', '黑桃9'],['黑桃10', '红桃10', '方块10', '梅花9', '黑桃9', '黑桃9'])[0])
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花9', '黑桃9', '黑桃9'],['黑桃10', '红桃10', '方块10', '梅花9', '黑桃9', '黑桃9'])[1])
    #print(rules.can_beat(['黑桃8', '红桃8', '方块8', '梅花9', '黑桃9', '黑桃9'],['黑桃10', '红桃10', '方块10', '梅花9', '黑桃9', '黑桃9'])[2])
    #print(rules.can_beat(['梅花6', '方块7', '红桃8', '红桃9', '梅花10'],['梅花6', '方块7', '红桃8', '红桃9', '梅花10', '红桃5']))
    
    #print(rules.get_play_value(['红桃10']))  #单牌 15
    #print(rules.get_play_value(['黑桃9', '红桃9', '方块9', '梅花9', '黑桃9', '红桃10'])) #炸弹 9
    #print(rules.get_play_value(['红桃10', '红桃9', '红桃8', '红桃J', '红桃10'])) #同花顺
    #print(rules.get_play_value(['红桃10', '黑桃10', '方块J', '梅花Q', '红桃10'])) #顺子
    #print(rules.get_play_value(['黑桃9', '黑桃10', '方块J', '梅花Q', '红桃10'])) #顺子
    #print(rules.get_play_value(['黑桃9', '黑桃9', '方块J', '红桃10', '红桃10'])) #三带二
    print(rules.get_play_value(['黑桃9', '黑桃9', '方块9', '梅花10', '红桃10', '红桃10'])) #钢板
    print(rules.get_play_value(['黑桃9', '黑桃9', '方块7', '梅花7', '红桃10', '红桃8'])) #木板
    print(rules.cards_to_mod_id(['黑桃9', '黑桃9', '方块9', '梅花10', '红桃10', '红桃10'])) #钢板
    print(rules.cards_to_mod_id(['黑桃9', '黑桃9', '方块7', '梅花7', '红桃10', '红桃8'])) #木板
    print(rules.cards_to_mod_id(['黑桃1', '黑桃2', '方块3', '梅花4', '红桃5', '红桃8','黑桃9', '黑桃9', '方块7', '梅花7', '红桃10', '红桃8'])) #木板
    

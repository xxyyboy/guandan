# 2025/3/16 17:16
import random
from rule import Rules
from give_cards import create_deck

class GuandanGame:
    def __init__(self, level_card=None):
        self.rules = Rules(level_card)  # 级牌
        self.players = [[] for _ in range(4)]  # 4 名玩家的手牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.deal_cards()  # 发牌

    def create_deck(self):
        """创建两副牌（不洗牌）"""
        SUITS = ['黑桃', '红桃', '梅花', '方块']
        RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [f"{suit}{rank}" for suit in SUITS for rank in RANKS] * 2
        deck += ['小王', '大王'] * 2  # 添加大小王
        return deck

    def deal_cards(self):
        """洗牌并发牌"""
        deck = self.create_deck()
        random.shuffle(deck)
        for i in range(len(deck)):
            self.players[i % 4].append(deck[i])
        for i in range(4):
            self.players[i].sort(key=lambda card: self.rules.get_rank(card), reverse=True)

    def play_turn(self):
        """当前玩家尝试出牌"""
        player_hand = self.players[self.current_player]

        # 如果所有人都 Pass 了（3 人 Pass）
        if self.pass_count == 3:
            print(f"\n🆕 3 人 Pass，本轮重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
            self.last_play = None  # 清空上一手牌
            self.pass_count = 0  # 重置 Pass 计数

        valid_moves = [card for card in player_hand if self.rules.is_valid_play([card])]

        if not valid_moves:
            print(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1  # 记录 Pass 次数
        else:
            # 选出最小的合法牌
            for card in valid_moves:
                if self.last_play is None or self.rules.can_beat(self.last_play, [card]):
                    self.last_play = [card]
                    self.last_player = self.current_player
                    player_hand.remove(card)
                    print(f"玩家 {self.current_player + 1} 出牌: {card}")

                    if not player_hand:
                        print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌，游戏结束！\n")
                        return True

                    self.pass_count = 0  # 有人出牌，重置 Pass 计数
                    break
            else:
                print(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1

        # 切换到下一个玩家
        self.current_player = (self.current_player + 1) % 4
        return False

    def play_game(self):
        """执行一整局游戏"""
        print("🎮 游戏开始！")
        while True:
            if self.play_turn():
                break


if __name__ == "__main__":
    game = GuandanGame(level_card="2")
    game.play_game()

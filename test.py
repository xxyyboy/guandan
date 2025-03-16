import random
from rule import Rules
from give_cards import create_deck, shuffle_deck, deal_cards

class GuandanGame:
    def __init__(self, level_card=None):
        self.rules = Rules(level_card)  # 级牌
        self.players = deal_cards(shuffle_deck(create_deck()))  # 直接调用 `give_cards.py` 的发牌逻辑
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数

    def play_turn(self):
        """当前玩家尝试出牌"""
        player_hand = self.players[self.current_player]

        if self.pass_count == 3:
            print(f"\n🆕 3 人 Pass，本轮重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
            self.last_play = None
            self.pass_count = 0

        valid_moves = [card for card in player_hand if self.rules.is_valid_play([card])]

        if not valid_moves:
            print(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
        else:
            for card in valid_moves:
                if self.last_play is None or self.rules.can_beat(self.last_play, [card]):
                    self.last_play = [card]
                    self.last_player = self.current_player
                    player_hand.remove(card)
                    print(f"玩家 {self.current_player + 1} 出牌: {card}")

                    if not player_hand:
                        print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌，游戏结束！\n")
                        return True

                    self.pass_count = 0
                    break
            else:
                print(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1

        self.current_player = (self.current_player + 1) % 4
        return False

    def play_game(self):
        """执行一整局游戏"""
        print("🎮 游戏开始！")
        while True:
            if self.play_turn():
                break


if __name__ == "__main__":
    game = GuandanGame(level_card=None)
    game.play_game()

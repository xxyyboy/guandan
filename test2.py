# 2025/3/21 更新提前结束规则
from give_cards import create_deck, shuffle_deck, deal_cards
from rule import Rules
import random


class GuandanGame:
    def __init__(self, level_card=None, user_player=None):
        self.rules = Rules(level_card)  # 级牌
        self.players = deal_cards(shuffle_deck(create_deck()))  # 发牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = user_player - 1 if user_player else None  # 转换为索引（0~3）
        self.ranking = []  # 存储出完牌的顺序

        # **手牌排序**
        for i in range(4):
            self.players[i] = self.sort_cards(self.players[i])

    def sort_cards(self, cards):
        """按牌的大小排序（从大到小）"""
        return sorted(cards, key=lambda card: self.rules.get_rank(card), reverse=True)

    def play_turn(self):
        """当前玩家尝试出牌"""
        player_hand = self.players[self.current_player]

        if self.user_player == self.current_player:
            self.show_user_hand()

        # **轮次重置逻辑**
        remaining_players = 4 - len(self.ranking)
        if self.pass_count == remaining_players - 1:
            print(f"\n🆕 {remaining_players - 1} 人 Pass，本轮重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
            self.last_play = None
            self.pass_count = 0

        if self.user_player == self.current_player:
            return self.user_play(player_hand)

        return self.ai_play(player_hand)

    def ai_play(self, player_hand):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        # **如果玩家已经打完，就跳过**
        if self.current_player in self.ranking:
            self.current_player = (self.current_player + 1) % 4
            return False

        # **构造可选牌型**
        possible_moves = []
        for size in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.rules.is_valid_play(move) and (not self.last_play or self.rules.can_beat(self.last_play, move)):
                    possible_moves.append(move)

        if not possible_moves:
            print(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
        else:
            chosen_move = random.choice(possible_moves)  # **随机选择一个合法的牌型**
            self.last_play = chosen_move
            self.last_player = self.current_player
            for card in chosen_move:
                player_hand.remove(card)
            print(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")

            if not player_hand:
                print(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.ranking.append(self.current_player)

            self.pass_count = 0

        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def user_play(self, player_hand):
        """让用户手动选择出牌"""
        while True:
            self.show_user_hand()
            choice = input("\n请选择要出的牌（用空格分隔），或直接回车跳过： ").strip()

            if choice == "":  # **回车等同于 pass**
                print(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                break

            if choice.lower() == "pass":
                print(f"玩家 {self.current_player + 1} Pass")
                self.pass_count += 1
                break

            selected_cards = choice.split()
            if all(card in player_hand for card in selected_cards) and self.rules.is_valid_play(selected_cards):
                if self.last_play is None or self.rules.can_beat(self.last_play, selected_cards):
                    for card in selected_cards:
                        player_hand.remove(card)
                    self.last_play = selected_cards
                    self.last_player = self.current_player
                    print(f"玩家 {self.current_player + 1} 出牌: {' '.join(selected_cards)}")

                    if not player_hand:
                        print(f"\n🎉 你出完所有牌！\n")
                        self.ranking.append(self.current_player)

                    self.pass_count = 0
                    break
                else:
                    print("❌ 你出的牌不能压过上一手牌，请重新选择！")
            else:
                print("❌ 你的输入无效，请确保牌在你的手牌中并符合规则！")

        self.current_player = (self.current_player + 1) % 4
        return self.check_game_over()

    def show_user_hand(self):
        """显示用户手牌（按排序后的顺序）"""
        sorted_hand = self.sort_cards(self.players[self.user_player])
        print("\n🃏 你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"🃏 场上最新出牌：{' '.join(self.last_play)}\n")

    def check_game_over(self):
        """检查游戏是否结束"""
        team_1 = {0, 2}
        team_2 = {1, 3}

        if len(self.ranking) >= 2:
            first_player, second_player = self.ranking[0], self.ranking[1]
            # **如果前两个出完牌的人是同一队伍，游戏提前结束**
            if (first_player in team_1 and second_player in team_1) or (first_player in team_2 and second_player in team_2):
                self.ranking.extend(i for i in range(4) if i not in self.ranking)  # 剩下的按出牌顺序补全
                self.show_game_result()
                return True

        if len(self.ranking) == 3:
            self.ranking.append(next(i for i in range(4) if i not in self.ranking))  # 找出最后一个玩家
            self.show_game_result()
            return True
        return False

    def show_game_result(self):
        """显示游戏结果"""
        team_1 = {0, 2}
        team_2 = {1, 3}
        first_player = self.ranking[0]

        if first_player in team_1:
            winning_team = "1 号和 3 号"
        else:
            winning_team = "2 号和 4 号"

        print("\n🎉 游戏结束！排名如下：")
        print(f"🥇 头游：玩家 {self.ranking[0] + 1}")
        print(f"🥈 二游：玩家 {self.ranking[1] + 1}")
        print(f"🥉 三游：玩家 {self.ranking[2] + 1}")
        print(f"🏅 末游：玩家 {self.ranking[3] + 1}")
        print(f"\n🏆 获胜队伍：{winning_team} 🎉🎉🎉")

    def play_game(self):
        """执行一整局游戏"""
        print("🎮 游戏开始！")
        while True:
            if self.play_turn():
                break


if __name__ == "__main__":
    user_pos = int(input("请选择你的座位（1~4）："))
    game = GuandanGame(level_card='2', user_player=user_pos)
    game.play_game()

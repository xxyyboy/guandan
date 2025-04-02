# 2025/3/16 17:16
# 2025/3/21 新增级牌升级规则
# 2025/4/1 完善规则
from give_cards import create_deck, shuffle_deck, deal_cards
from rule import Rules
import random
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

class Player:
    def __init__(self, hand):
        """
        程序里的玩家是从0开始的，输出时会+1
        """
        self.hand = hand  # 手牌
        self.played_cards = []  # 记录已出的牌

class GuandanGame:
    def __init__(self, user_player=None, active_level=None,verbose=True , print_history=False):
        # **两队各自的级牌**
        self.print_history = print_history
        self.active_level = active_level if active_level else random.choice(range(2, 15))
        # 历史记录，记录最近 20 轮的出牌情况（每轮包含 4 个玩家的出牌）
        self.history = []
        # **只传当前局的有效级牌**
        self.rules = Rules(self.active_level)
        # self.players = deal_cards(shuffle_deck(create_deck()))
        self.players = [Player(hand) for hand in deal_cards(shuffle_deck(create_deck()))]# 发牌
        self.current_player = 0  # 当前出牌玩家
        self.last_play = None  # 记录上一手牌
        self.last_player = -1  # 记录上一手是谁出的
        self.pass_count = 0  # 记录连续 Pass 的次数
        self.user_player = user_player - 1 if user_player else None  # 转换为索引（0~3）
        self.ranking = []  # 存储出完牌的顺序
        self.recent_actions = {i: [] for i in range(4)}
        self.verbose = verbose  # 控制是否输出文本
        self.team_1 = {0, 2}
        self.team_2 = {1, 3}

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
        active_players = sum(1 for p in self.players if len(p.hand) > 0)

        # **如果 Pass 的人 == "当前有手牌的玩家数 - 1"，就重置轮次**
        if self.pass_count >= (active_players - 1):
            self.log(f"\n🆕 {self.pass_count} 人 Pass，轮次重置！玩家 {self.current_player + 1} 可以自由出牌。\n")
            self.last_play = None  # ✅ 允许新的自由出牌
            self.pass_count = 0  # ✅ Pass 计数归零

        result = self.ai_play(player)
        '''
        # **AI 或用户出牌**
        if self.user_player == self.current_player:
            result = self.user_play(player)
        else:
            result = self.ai_play(player)
        '''
        # **记录最近 5 轮历史**
        if self.current_player == 0:
            round_history = [self.recent_actions[i] for i in range(4)]
            self.history.append(round_history)
            self.recent_actions=[['None'],['None'],['None'],['None']]
            '''
            if len(self.history) > 20:
                self.history.pop(0)
            '''
        return result

    def ai_play(self, player):
        """AI 出牌逻辑（随机选择合法且能压过上家的出牌）"""

        # **如果玩家已经打完，仍然记录一个空列表，然后跳过**
        if self.current_player in self.ranking:
            self.recent_actions[self.current_player] = []  # 记录空列表
            self.current_player = (self.current_player + 1) % 4

            return self.check_game_over()

        player_hand = player.hand

        # **构造可选牌型**
        possible_moves = []
        for size in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(len(player_hand) - size + 1):
                move = player_hand[i:i + size]
                if self.rules.is_valid_play(move) and (not self.last_play or self.rules.can_beat(self.last_play, move)):
                    possible_moves.append(move)

        if not possible_moves:
            self.log(f"玩家 {self.current_player + 1} Pass")
            self.pass_count += 1
            self.recent_actions[self.current_player] = ['Pass']  # **记录 Pass**
        else:
            chosen_move = random.choice(possible_moves)  # **随机选择一个合法的牌型**
            self.last_play = chosen_move
            self.last_player = self.current_player
            for card in chosen_move:
                player_hand.remove(card)
            self.log(f"玩家 {self.current_player + 1} 出牌: {' '.join(chosen_move)}")
            self.recent_actions[self.current_player] = list(chosen_move)  # **记录出牌**

            if not player_hand:  # **玩家出完牌**
                self.log(f"\n🎉 玩家 {self.current_player + 1} 出完所有牌！\n")
                self.ranking.append(self.current_player)
                self.recent_actions[self.current_player] = list(chosen_move)  # **记录出牌**

            self.pass_count = 0

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
                return True

        # **如果 3 人出完了，自动补全最后一名，游戏结束**
        if len(self.ranking) == 3:
            self.ranking.append(next(i for i in range(4) if i not in self.ranking))  # 找出最后一个玩家
            self.update_level()
            return True

        return False

    def update_level(self):
        """升级级牌"""
        first_player = self.ranking[0]  # 第一个打完牌的玩家
        winning_team = 1 if first_player in self.team_1 else 2
        # 确定队友
        teammate = 2 if first_player == 0 else 0 if first_player == 2 else 3 if first_player == 1 else 1

        # 找到队友在排名中的位置
        teammate_position = self.ranking.index(teammate)

        # 头游 + 队友的名次，确定得分
        upgrade_map = {1: 3, 2: 2, 3: 1}  # 头游 + (队友的名次) 对应的升级规则
        upgrade_amount = upgrade_map[teammate_position]

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
        sorted_hand = self.sort_cards(self.players[self.user_player])
        print("\n🃏 你的手牌：", " ".join(sorted_hand))
        if self.last_play:
            print(f"🃏 场上最新出牌：{' '.join(self.last_play)}\n")

if __name__ == "__main__":
    game = GuandanGame(active_level=None,verbose=True,print_history=True)
    game.play_game()


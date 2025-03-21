# 2025/3/21 15:17
import gym
from gym import spaces
import numpy as np
from Game import GuandanGame

class GuandanEnv(gym.Env):
    def __init__(self):
        super(GuandanEnv, self).__init__()
        self.game = GuandanGame(level_card='2')
        self.action_space = spaces.Discrete(54)  # 假设最多 54 种出牌方式
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,), dtype=np.float32)

    def reset(self):
        self.game = GuandanGame(level_card='2')
        return self._get_obs()



    def _get_obs(self):
        obs = np.zeros(108 + 108 + (108 * 4) + (108 * 3) + (27 * 3) + 13 + (108 * 4 * 5) + 3 + 3 + 3)

        # 1. 当前玩家的手牌
        for card in self.game.players[self.game.current_player].hand:
            obs[self.card_to_index(card)] = 1

        # 2. 其他玩家的手牌信息（可以用模糊表示，比如数量）
        offset = 108
        for i, player in enumerate(self.game.players):
            if i != self.game.current_player:
                for card in player.hand:
                    obs[offset + self.card_to_index(card)] = 1
                offset += 108

        # 3. 每个玩家最近动作
        offset += 108
        for i, action in enumerate(self.game.recent_actions):
            for card in action:
                obs[offset + i * 108 + self.card_to_index(card)] = 1

        # 4. 其他玩家的已出牌
        offset += 108 * 4
        for i, player in enumerate(self.game.players):
            if i != self.game.current_player:
                for card in player.played_cards:
                    obs[offset + i * 108 + self.card_to_index(card)] = 1

        # 5. 其他玩家剩余牌数
        offset += 108 * 3
        for i, player in enumerate(self.game.players):
            if i != self.game.current_player:
                obs[offset + i * 27 + min(player.hand_size, 26)] = 1  # 限制最大 26 张
        offset += 27 * 3

        # 6. 当前级牌
        obs[offset + self.level_card_to_index(self.game.level_card)] = 1
        offset += 13

        # 7. 最近 20 轮出牌历史
        for round_idx, round_actions in enumerate(self.game.history[-5:]):  # 只取最近 5 轮
            for player_idx, action in enumerate(round_actions):
                for card in action:
                    obs[offset + round_idx * 108 * 4 + player_idx * 108 + self.card_to_index(card)] = 1
        offset += 108 * 4 * 5

        # 8. 协作/压制/辅助状态（可以基于比赛逻辑计算）
        obs[offset] = self.compute_coop_status()
        obs[offset + 1] = self.compute_dwarf_status()
        obs[offset + 2] = self.compute_assist_status()

        return obs

    def step(self, action):
        """
        执行一个动作，并返回新的状态、奖励、是否结束、以及额外信息。
        """
        current_player = self.game.current_player
        player = self.game.players[current_player]

        # 确保行动合法
        if not self.rules.is_valid_play(player.hand, action):
            return self._get_obs(), -1, False, {"error": "Invalid action"}  # 非法行动，惩罚 -1

        # 1. 执行出牌
        for card in action:
            player.hand.remove(card)
            player.played_cards.append(card)  # 记录已出的牌
        self.game.recent_actions[current_player] = action  # 记录最近的动作

        # 2. 处理逢人配（红桃级牌）—— 在合法性检测时已处理，这里不需要特别修改

        # 3. 计算奖励
        reward = 0.1  # 每次出牌都给予基础奖励

        # **如果该玩家出完所有牌**
        if len(player.hand) == 0:
            player.finished = True
            reward += 1  # 额外奖励
            print(f"玩家 {current_player} 已打完")

        # 4. 判断是否游戏结束（新的规则）
        active_players = [p for p in self.game.players if not p.finished]

        if len(active_players) == 1:  # 只剩 1 人未出完牌，游戏结束
            self.game_done = True
            winning_team = 0 if current_player in [0, 2] else 1  # 0-2 队 vs 1-3 队
            print(f"游戏结束！胜利队伍是 {'0-2 队' if winning_team == 0 else '1-3 队'}")

            # 胜利队伍奖励
            for i, p in enumerate(self.game.players):
                if i in [0, 2] and winning_team == 0:
                    reward = 1  # 队伍 0-2 胜利
                elif i in [1, 3] and winning_team == 1:
                    reward = 1  # 队伍 1-3 胜利
                else:
                    reward = -1  # 失败者惩罚

        # 5. 切换到下一个玩家（如果游戏没结束）
        if not self.game_done:
            self._next_player()

        return self._get_obs(), reward, self.game_done, {}

    def render(self, mode="human"):
        self.game.show_user_hand()

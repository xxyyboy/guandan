# 2025/5/13 15:45
import json
import matplotlib.pyplot as plt

with open("doudizhu_actions.json", "r", encoding="utf-8") as f:
    M = json.load(f)
action_dim = len(M)
# 构建动作映射字典
M_id_dict = {a['id']: a for a in M}
# 奖励计算参数
lambda_penalty = 0.8
reward_list = []

for action_id in range(len(M)):
    entry = M_id_dict[action_id]
    points = entry["points"]
    logic_point = entry["logic_point"]

    if action_id == 0:  # pass无奖励
        reward = 0
    else:
        if 120 <= action_id < 330:
            # DONE: 三带中对子的判断
            side_point = entry['points'][-1]  # 直接取副牌点数
            alpha = 0.8  # 控制副牌惩罚强度（越大惩罚越重）
            reward = len(entry['points']) / ((entry['logic_point'] ** 0.5) + alpha * (side_point ** 0.5))
        else:
            reward = float(len(entry['points']) / (entry['logic_point'] ** 0.5))
        # 对高级出牌奖励修正
        if 120 <= action_id < 330: reward += reward *2
        if action_id == 119: reward += 1

    reward_list.append(reward)

# 绘图
plt.figure(figsize=(12, 5))
plt.plot(reward_list, label="Reward by Action ID")
plt.xlabel("Action ID")
plt.ylabel("Reward")
plt.title("Reward Distribution Across Action IDs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def test():
    memory=[[['梅花8'], ['Pass'], ['红桃K'], ['小王']],
    [['Pass'], ['方块7', '梅花7', '红桃7', '梅花7', '黑桃7'], ['Pass'], ['Pass']],
    [['Pass'], ['梅花K', '方块K'], ['红桃8', '黑桃8', '红桃8', '方块8'], ['Pass']],
    [['Pass'], ['Pass'], ['方块9'], ['红桃J']],
    [['Pass'], ['方块A'], ['Pass'], ['方块10', '梅花10', '梅花10', '黑桃10']],
    [['Pass'], ['Pass'], ['Pass'], ['梅花Q']],
    [['梅花A'], ['Pass'], ['Pass'], ['梅花6', '方块6', '方块6', '黑桃6']],
    [['Pass'], ['Pass'], ['Pass'], ['方块4']],
    [['方块7'], ['黑桃10'], ['红桃K'], ['黑桃A']],
    [['大王'], ['Pass'], ['Pass'], ['Pass']],
    [['梅花3', '方块3'], ['梅花9', '红桃9'], ['梅花J', '黑桃J'], ['黑桃Q', '红桃Q']],
    [['Pass'], ['梅花2', '方块2', '方块2', '红桃2'], ['Pass'], ['Pass']],
    [['Pass'], ['方块3'], ['方块5'], ['黑桃Q']],
    [['Pass'], ['红桃A'], ['Pass'], ['Pass']],
    [['Pass'], ['黑桃A'], ['Pass'], ['Pass']],
    [['Pass'], ['红桃A'], ['Pass'], ['Pass']],
    [['小王'], ['Pass'], ['Pass'], ['Pass']],
    [['方块5'], ['梅花Q'], ['Pass'], ['梅花K']],
    [['Pass'], ['大王'], ['Pass'], ['Pass']],
    [['Pass'], ['红桃3'], ['方块Q'], ['方块A']],
    [['Pass'], ['Pass'], ['Pass'], ['红桃J', '方块J']],
    [['Pass'], ['Pass'], ['Pass'], ['方块4']],
    [['黑桃9'], ['Pass'], ['梅花J'], ['Pass']],
    [['方块Q'], ['Pass'], ['梅花A'], ['Pass']],
    [['Pass'], ['Pass'], ['黑桃4', '梅花4', '红桃4'], ['Pass']],
    [['Pass'], ['Pass'], ['黑桃2'], ['黑桃5']],
    [['红桃10'], ['方块K'], ['Pass'], ['Pass']],
    [['Pass'], ['黑桃8'], ['梅花9'], ['Pass']],
    [['黑桃J'], ['Pass'], ['红桃Q'], ['Pass']],
    [['Pass'], ['Pass'], ['方块9'], ['Pass']],
    [['红桃10'], ['Pass'], ['黑桃K'], ['Pass']],
    [['Pass'], ['Pass'], ['黑桃3'], ['黑桃5']],
    [['方块J'], ['Pass'], ['Pass'], ['Pass']],
    [['红桃9', '黑桃9'], ['Pass'], ['Pass'], ['Pass']],
    [['红桃5'], ['红桃6'], ['Pass'], ['Pass']],
    [['Pass'], ['红桃4'], ['红桃6'], ['Pass']],
    [['Pass'], ['Pass'], ['红桃7'], ['Pass']],
    [['梅花8'], ['Pass'], ['Pass'], ['Pass']],
    [['黑桃K'], ['Pass'], ['Pass'], ['Pass']],
    [['方块8'], ['Pass'], ['方块10'], ['Pass']],
    [['Pass'], ['Pass'], ['黑桃6'], ['Pass']],
    [['Pass'], ['Pass'], ['梅花5'], ['Pass']],
    [['黑桃7'], ['Pass'], [], ['Pass']],
    [['黑桃4'], ['红桃5'], [], ['Pass']],
    [['黑桃3', '梅花3'], [], [], ['Pass']],
    [['梅花6'], [], [], ['Pass']],
    [['红桃2', '黑桃2'], [], [], ['Pass']],
    [['梅花5'], [], [], ['Pass']],
    [['梅花4'], ['None'], ['None'], ['None']]]
    for i, s in enumerate(memory):
        print( 0.9 ** (len(memory) - i - 1) * 3)
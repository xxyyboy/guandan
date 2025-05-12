# [掼蛋强化学习](https://guandan.streamlit.app/)

![Static Badge](https://img.shields.io/badge/ver.-1.2.1-E85889)
![GitHub](https://img.shields.io/github/license/746505972/guandan?logo=github)
<div style="margin-top: 6px; background: #f1f1f1; padding: 1px;
            border-radius: 6px; color: #333; font-size: 14px;">
</div>
## 2025/4/25
### 建立A2C网络，能够训练、预测。

- `ActorNet` 输出结构动作 `action_id` 的概率分布（考虑了 mask）；

- `CriticNet` 输出当前 `state` 的 `value` 估计；

### 奖励函数
```
r=reward + gamma ** (len(memory) - i - 1) * final_reward
```
- `reward`为即时奖励，值为出牌长度*牌型的`logic_point`

- `final_reward`为整局奖励，队伍获胜为正，12名为3，13名为2，14名为1，反之亦然

- `memory`是一局的记录，结构为`[{state,action_id,reward},{state,action_id,reward},...]`

### 优势函数：

$advantage=r+\gamma*V(state')-V(state)$

其中最后一个 $transition$ 没有 $state'$ ,设 $\gamma*V(state')=0$

`Actor Loss`= $-log(p)*advantage$

`Critic Loss` = $||advantage||^2$
## 2025/4/20
108维具体参照：
`
{[黑桃2-A]+[红桃2-A]+[梅花2-A]+[方块2-A]}*2+[小王,大王]*2
`
>e.g.
> 
>hand = ['红桃2', '红桃2','黑桃3', '红桃3', '黑桃4', '红桃4', '红桃4', '黑桃4','黑桃5', '红桃5','大王','小王','大王']
> 
> obs:
> 
> [0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1.]

**状态表示**

**使用 One-Hot 编码（或多热编码）来表示手牌和其他信息**

| 特征        | 维度          | 说明                     |
|-----------|-------------|------------------------|
| 当前玩家的手牌   | 108         | 54 张牌，每张牌是否在手（双份）      |
| 其他玩家的手牌   | 3           | 3 维，表示剩余手牌数量（归一化））     |
| 每个玩家最近动作  | 108 × 4     | 每个玩家的最近出牌              |
| 其他玩家出的牌   | 108 × 3     | 记录已经打出的牌               |
| 当前级牌      | 13          | 级牌 one-hot 表示          |
| 最近 20 次动作 | 108 × 4 × 5 | 5 轮历史，每轮 4 玩家，每人 108 维 |
| 协作状态      | 3           | 标识与队友的配合程度             |
| 压制状态      | 3           | 标识对敌人的打压情况             |
| 辅助状态      | 3           | 标识是否有意给队友铺路            |
| 总维度       | `3049`      |                        |

![graphviz.png](graphviz.png)
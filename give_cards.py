import random

# 定义牌型
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# 创建两副牌
def create_deck():
    deck = []
    for _ in range(2):  # 两副牌
        for suit in SUITS:
            for rank in RANKS:
                deck.append(rank + suit)
        # 添加大小王
        deck.append('小王')
        deck.append('大王')
    return deck

# 洗牌
def shuffle_deck(deck):
    random.shuffle(deck)
    return deck

# 发牌
def deal_cards(deck):
    players = [[], [], [], []]  # 4个玩家
    for i in range(len(deck)):
        players[i % 4].append(deck[i])
    return players

if __name__ == "__main__":
    # 创建牌堆
    deck = create_deck()
    
    # 洗牌
    shuffled_deck = shuffle_deck(deck)
    
    # 发牌
    players = deal_cards(shuffled_deck)
    
    # 打印每个玩家的牌
    for i, player in enumerate(players):
        print(f"玩家{i+1}的牌：")
        print(sorted(player))
        print()

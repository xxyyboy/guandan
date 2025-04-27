# app.py
import streamlit as st
import time
from display import GuandanGame

# 初始化游戏
if 'game' not in st.session_state:
    st.session_state.game = GuandanGame(user_player=1, active_level=None, verbose=False, print_history=False)
    st.session_state.selected_cards = []
    st.session_state.logs = []
    st.session_state.game_over = False

game = st.session_state.game

# 页面设置
st.set_page_config(page_title="掼蛋 AI", layout="wide")
st.title("掼蛋 AI 对局 - 玩家1 视角")

# 显示其他玩家剩余手牌
st.subheader("其他玩家手牌数")
cols = st.columns(3)
for i in range(1, 4):
    cols[i-1].metric(f"玩家{i+1}", f"{len(game.players[i].hand)} 张")

# 显示最近出牌
st.subheader("最近出牌")
cols = st.columns(4)
for i in range(4):
    last_play = game.players[i].last_played_cards
    play_text = "Pass" if last_play == ['Pass'] else " ".join(last_play) if last_play else "无"
    cols[i].info(f"玩家{i+1}: {play_text}")

# 显示场上最新牌
st.subheader("场上最新出牌")
if game.last_play:
    st.success(f"玩家 {game.last_player+1} 出了: {' '.join(game.last_play)}")
else:
    st.info("当前是自由出牌回合（必须主动出牌）")

# --- 当前玩家操作区 ---
hand = game.players[game.user_player].hand
is_my_turn = (game.current_player == game.user_player) and not st.session_state.game_over


st.subheader("你的手牌")
if is_my_turn:
    selected = st.multiselect(
        "选择要打出的牌：",
        options=list(hand),
        default=[],
        key="selected_cards"
    )

    st.session_state.selected_cards = selected
    st.write(f"✅ 当前选中：{' '.join(selected)}")

    # 出牌 + PASS按钮
    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("🎴 打出选择的牌", type="primary", disabled=not st.session_state.selected_cards):
            try:
                game.user_submit_play(st.session_state.selected_cards)
                st.session_state.logs.append(f"你出牌：{' '.join(st.session_state.selected_cards)}")
                st.session_state.selected_cards.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ 出牌失败：{str(e)}")

    with action_col2:
        if not game.is_free_turn:
            if st.button("🚫 PASS (不出)", key="pass_button_enabled"):
                try:
                    game.user_submit_pass()
                    st.session_state.logs.append("你选择PASS")
                    st.session_state.selected_cards.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ PASS失败：{str(e)}")
        else:
            st.warning("当前是自由出牌回合，必须主动出牌！")
            st.button("🚫 PASS (不出)", disabled=True, key="pass_button_disabled")

    # AI出牌建议
    st.subheader("AI给出的出牌建议")
    suggestions = game.get_ai_suggestions()
    for s in suggestions:
        st.info(s)

else:
    st.info("等待AI玩家出牌中...")

    # 后端已经处理了AI出牌 advance_turn
    time.sleep(0.5)
    if not st.session_state.game_over:
        st.rerun()

# 游戏日志
st.subheader("游戏日志")
log_container = st.container(height=300)
with log_container:
    for log in reversed(st.session_state.logs):
        st.write(log)

# 游戏结束
if st.session_state.game_over:
    st.success(f"🏆 游戏结束！{game.winning_team}号队伍胜利，升 {game.upgrade_amount} 级！")
    st.write("最终排名：")
    ranks = ["头游", "二游", "三游", "末游"]
    for i, player in enumerate(game.ranking):
        st.write(f"{ranks[i]}: 玩家{player+1}")

    if st.button("开始新的一局"):
        st.session_state.clear()
        st.rerun()

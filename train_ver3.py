import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from env import BatteryEnv
from agent import QAgent, quantize_state

# 논문용 고해상도 및 화이트 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 0  # 제목 공간 자동 제거

def get_next_save_dir(base_name="conference_final_selection"):
    if not os.path.exists(base_name): return base_name
    i = 1
    while os.path.exists(f"{base_name}_{i}"): i += 1
    return f"{base_name}_{i}"

def moving_average(data, window=300):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

# --- 비교 정책군 함수 ---
def run_predictive_policy(episodes=4000):
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    avg_unplug = 450.0; safety_margin = 70.0
    for _ in range(episodes):
        env.reset(); done = False; ep_reward = 0
        start_charge_time = max(0, avg_unplug - 120 - safety_margin)
        while not done:
            action = 1 if env.current_time >= start_charge_time else 0
            _, reward, done, info = env.step(action)
            ep_reward += reward
        avg_unplug = 0.8 * avg_unplug + 0.2 * env.actual_unplug_time
        rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
    return rewards, idles, final_socs

def run_greedy_policy(episodes=4000):
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    for _ in range(episodes):
        env.reset(); done = False; ep_reward = 0
        while not done:
            _, reward, done, info = env.step(1)
            ep_reward += reward
        rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
    return rewards, idles, final_socs

def train_qlearning_full(env, agent, episodes=4000):
    rewards, idles, final_socs = [], [], []
    for ep in range(episodes):
        state = env.reset(); done = False; ep_reward = 0
        while not done:
            state_disc = quantize_state(state)
            action = agent.choose_action(state_disc)
            next_state, reward, done, info = env.step(action)
            agent.learn(state_disc, action, reward, quantize_state(next_state), ep)
            state = next_state
            ep_reward += reward
        agent.decay_epsilon()
        rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
    return rewards, idles, final_socs

if __name__ == "__main__":
    EPISODES = 4000
    NUM_RUNS = 10
    save_dir = get_next_save_dir()
    os.makedirs(save_dir, exist_ok=True)

    # 차별화된 학회지용 색상 코드
    c_q = '#1A4D80' # Navy (Proposed)
    c_p = '#C47327' # Amber (Baseline)
    c_g = '#7F8C8D' # Grey (Greedy)

    print(f"총 {NUM_RUNS}회 독립 실행 및 다변화 그래프 생성을 시작합니다.")

    for run in range(1, NUM_RUNS + 1):
        print(f"Run [{run}/{NUM_RUNS}] 진행 및 그래프 저장 중...")
        env = BatteryEnv(); agent = QAgent()
        
        # 데이터 획득
        q_rew, q_idl, q_soc = train_qlearning_full(env, agent, EPISODES)
        p_rew, p_idl, p_soc = run_predictive_policy(EPISODES)
        g_rew, g_idl, g_soc = run_greedy_policy(EPISODES)

        eval_window = 500
        q_mv = moving_average(q_rew)
        p_mv = moving_average(p_rew)
        g_mv = moving_average(g_rew)
        x_range = np.arange(len(q_mv))

        # --- 1-{run} Reward Plot (Shape: Area fill + Thick line) ---
        plt.figure(figsize=(9, 5.5))
        plt.fill_between(x_range, q_mv, color=c_q, alpha=0.1) # 배경 채우기로 형태 차별화
        plt.plot(x_range, q_mv, label="Q-Learning", color=c_q, linewidth=2)
        plt.plot(moving_average(p_rew), label="EMA Model", color=c_p, linewidth=1.2, linestyle='--')
        plt.plot(moving_average(g_rew), label="Greedy", color=c_g, linewidth=1, linestyle=':', alpha=0.6)
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Reward (w=300)")
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"1-{run}_Reward.png"), dpi=300)
        plt.close()

        # --- 2-{run} Aging Plot (Shape: Dashed path + Sampling markers) ---
        q_idl_mv = moving_average(q_idl)
        p_idl_mv = moving_average(p_idl)
        plt.figure(figsize=(9, 5.5))
        plt.plot(q_idl_mv, color=c_q, linewidth=2, label="Q-Learning")
        # 500 에피소드마다 마커를 찍어 선의 형태를 다르게 표현
        markers_x = np.arange(0, len(q_idl_mv), 500)
        plt.scatter(markers_x, q_idl_mv[markers_x], color=c_q, s=50, edgecolors='black', zorder=5)
        
        plt.plot(moving_average(p_idl), color=c_p, linewidth=1.5, linestyle='-.', label="EMA Model")
        plt.plot(moving_average(g_idl), color=c_g, linewidth=1, linestyle=':', alpha=0.6, label="Greedy")
        
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Idle Time [min]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"2-{run}_Aging.png"), dpi=300)
        plt.close()

        # --- 3-{run} Trade-off Scatter (Shape: Target Icon + Grid layout) ---
        plt.figure(figsize=(8, 8))
        q_i_fin, q_s_fin = np.mean(q_idl[-eval_window:]), np.mean(q_soc[-eval_window:])
        p_i_fin, p_s_fin = np.mean(p_idl[-eval_window:]), np.mean(p_soc[-eval_window:])
        g_i_fin, g_s_fin = np.mean(g_idl[-eval_window:]), np.mean(g_soc[-eval_window:])

        # 육각형, 별형, 대각선 등 신규 마커 적용
        plt.scatter(q_i_fin, q_s_fin, color=c_q, marker='H', s=250, label="Q-Learning", edgecolors='black', linewidths=1.5, zorder=5)
        plt.scatter(p_i_fin, p_s_fin, color=c_p, marker='p', s=200, label="EMA Model", edgecolors='black', zorder=4)
        plt.scatter(g_i_fin, g_s_fin, color=c_g, marker='X', s=200, label="Greedy", edgecolors='black', zorder=3)

        # 타겟 가이드라인 추가
        plt.axhline(100, color='black', linewidth=0.8, linestyle='--')
        plt.axvspan(-10, 65, color='#F0F7FF', alpha=0.5, label='Optimal Protection Zone')

        plt.xlabel("Average Overcharge Idle Time [min] (← Lower is Better)")
        plt.ylabel("Average Final SoC [%] (↑ Higher is Better)")
        plt.xlim(450, -10); plt.ylim(70, 105)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"3-{run}_Tradeoff.png"), dpi=300)
        plt.close()

        # --- 4-{run} Boxplot (Shape: Stacked Vertical layout + Horizontal boxes) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
        
        # 최종 SoC 안정성 (가로형 박스플롯으로 변환)
        sns.boxplot(data=[q_soc[-eval_window:], p_soc[-eval_window:]], 
                    palette=[c_q, c_p], orient='h', width=0.5, ax=ax1)
        ax1.set_yticklabels(['Q-Learning', 'EMA Model'], fontweight='bold')
        ax1.set_xlabel('Final SoC [%]')
        
        # 방치 시간 안정성 (가로형 박스플롯으로 변환)
        sns.boxplot(data=[q_idl[-eval_window:], p_idl[-eval_window:]], 
                    palette=[c_q, c_p], orient='h', width=0.5, ax=ax2)
        ax2.set_yticklabels(['Q-Learning', 'EMA Model'], fontweight='bold')
        ax2.set_xlabel('Overcharge Idle Time [min]')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"4-{run}_Stability.png"), dpi=300)
        plt.close()

    print(f"\n[성공] 10회의 독립 시뮬레이션 결과가 '{save_dir}' 폴더에 생성되었습니다. 제목이 제거된 가장 선명한 그래프를 선택하여 사용하십시오.")
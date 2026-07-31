import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from env import BatteryEnv
from agent import QAgent, quantize_state

# 학술적인 레이아웃 테마 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def get_next_save_dir(base_name="conference_results"):
    if not os.path.exists(base_name): return base_name
    i = 1
    while os.path.exists(f"{base_name}_{i}"): i += 1
    return f"{base_name}_{i}"

def moving_average(data, window=300): # 논문 설명글의 w=300 윈도우 크기 반영
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def run_predictive_policy(episodes=4000):
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    avg_unplug = 450.0 
    safety_margin = 70.0
    
    for _ in range(episodes):
        env.reset() 
        done = False; ep_reward = 0
        start_charge_time = max(0, avg_unplug - 120 - safety_margin)
        
        while not done:
            action = 1 if env.current_time >= start_charge_time else 0
            _, reward, done, info = env.step(action)
            ep_reward += reward
            
        actual = env.actual_unplug_time
        avg_unplug = 0.8 * avg_unplug + 0.2 * actual
        rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
        
    return rewards, idles, final_socs

def run_greedy_policy(episodes=4000):
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    for _ in range(episodes):
        env.reset(); done = False; ep_reward = 0
        while not done:
            action = 1
            _, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
    return rewards, idles, final_socs

def train_qlearning_full(env, agent, episodes=4000):
    rewards, idles, final_socs = [], [], []
    print(f"{'Episode':<10} | {'Total Reward':<15} | {'Final SoC':<10}")
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
        
        if (ep + 1) % 500 == 0:
            print(f"{ep+1:<10} | {ep_reward:<15.2f} | {info['soc']:<10.1f}%")
            
    return rewards, idles, final_socs

if __name__ == "__main__":
    EPISODES = 4000
    NUM_RUNS = 10
    save_dir = get_next_save_dir()
    os.makedirs(save_dir, exist_ok=True)

    # 신규 차별화 컬러 파레트 정의 (Deep Navy, Warm Amber, Slate Gray)
    colors = {
        'Q-Learning': '#0F4C81',     # 제안 기법: 클래식 딥네이비 (신뢰감 부여)
        'Predictive ML': '#E08A3C',   # 비교 기법 1: 웜 앰버 오렌지
        'Greedy': '#7F8C8D'          # 비교 기법 2: 슬레이트 그레이
    }

    print(f"학회 제출용 그래프 생성 프로그램을 시작합니다. 저장 경로: {save_dir}")

    for run in range(1, NUM_RUNS + 1):
        print(f"\n[{run}/{NUM_RUNS}] 차별화 시뮬레이션 Run 구동 중...")
        env = BatteryEnv(); agent = QAgent()
        
        q_rew, q_idl, q_soc = train_qlearning_full(env, agent, EPISODES)
        p_rew, p_idl, p_soc = run_predictive_policy(EPISODES) 
        g_rew, g_idl, g_soc = run_greedy_policy(EPISODES)

        eval_window = 500 

        # --- 1. Reward Convergence (학회 버전 디자인) ---
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(moving_average(q_rew), label="Q-Learning (Proposed)", color=colors['Q-Learning'], linewidth=2.2, zorder=4)
        ax.plot(moving_average(p_rew), label="Predictive ML (Baseline)", color=colors['Predictive ML'], linewidth=1.8, linestyle='--', alpha=0.9, zorder=3)
        ax.plot(moving_average(g_rew), label="Greedy Control", color=colors['Greedy'], linewidth=1.5, linestyle=':', alpha=0.7, zorder=2)
        
        ax.set_title("Learning Curve: Reward Convergence (Conference Ver.)", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Episode", fontsize=11, labelpad=8)
        ax.set_ylabel("Moving Average of Reward (w=300)", fontsize=11, labelpad=8)
        ax.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='#EAEAEA')
        ax.grid(True, linestyle='-', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"1-{run}_Reward_Conf.png"), dpi=300)
        plt.close()

        # --- 2. Overcharge Time Reduction (학회 버전 디자인) ---
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(moving_average(q_idl), label="Q-Learning (Proposed)", color=colors['Q-Learning'], linewidth=2.2, zorder=4)
        ax.plot(moving_average(p_idl), label="Predictive ML (Baseline)", color=colors['Predictive ML'], linewidth=1.8, linestyle='--', alpha=0.9, zorder=3)
        ax.plot(moving_average(g_idl), label="Greedy Control", color=colors['Greedy'], linewidth=1.5, linestyle=':', alpha=0.7, zorder=2)
        
        # 제안 기법 최적 성능 수렴 지점 강조용 은은한 음영 추가
        ax.axhspan(0, 65, color='#F0F4F8', alpha=0.5, zorder=1)
        
        ax.set_title("Battery Protection: Overcharge Time Reduction (Conference Ver.)", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Episode", fontsize=11, labelpad=8)
        ax.set_ylabel("Moving Average of Idle Time [min] (w=300)", fontsize=11, labelpad=8)
        ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='#EAEAEA')
        ax.grid(True, linestyle='-', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"2-{run}_Aging_Conf.png"), dpi=300)
        plt.close()

        # --- 3. Performance Trade-off Scatter Plot (학회 버전 최적화) ---
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        
        # 완전한 마커 형태 다변화 (원, 다이아몬드, 역삼각형) 및 외곽선 스타일링
        policies = {
            "Q-Learning": (np.mean(q_idl[-eval_window:]), np.mean(q_soc[-eval_window:]), colors['Q-Learning'], 'o', 180),
            "Predictive ML": (np.mean(p_idl[-eval_window:]), np.mean(p_soc[-eval_window:]), colors['Predictive ML'], 'D', 140),
            "Greedy": (np.mean(g_idl[-eval_window:]), np.mean(g_soc[-eval_window:]), colors['Greedy'], 'v', 140)
        }

        for name, (idle_avg, soc_avg, color, marker, size) in policies.items():
            ax.scatter(idle_avg, soc_avg, color=color, marker=marker, s=size, label=name, edgecolors='#333333', linewidths=1.2, zorder=5)
            ax.annotate(f" {name}", (idle_avg, soc_avg), xytext=(8, -2), textcoords='offset points', 
                        fontsize=11, color='#222222', fontweight='bold', verticalalignment='center')

        ax.axhline(100.0, color='#999999', linestyle='-.', alpha=0.6)
        ax.set_title("Performance Trade-off Analysis (Conference Ver.)", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Average Overcharge Idle Time [min] (Lower is Better ←)", fontsize=11, labelpad=8)
        ax.set_ylabel("Average Final SoC [%] (Higher is Better ↑)", fontsize=11, labelpad=8)
        
        max_idle_time = max(max(g_idl[-eval_window:]), max(p_idl[-eval_window:]))
        ax.set_xlim(max(max_idle_time, 350), -10) 
        ax.set_ylim(70, 105) 
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"3-{run}_Tradeoff_Scatter_Conf.png"), dpi=300)
        plt.close()

        # --- 4. Robustness Box Plot (학회 전용 투톤 콤팩트 스타일) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
        
        # 4-1. SoC 안정성 플롯 개편
        sns.boxplot(
            data=[q_soc[-eval_window:], p_soc[-eval_window:]], 
            palette=[colors['Q-Learning'], colors['Predictive ML']],
            width=0.5, fliersize=4, linewidth=1.2, ax=ax1
        )
        ax1.set_xticklabels(['Q-Learning', 'Predictive ML'], fontsize=10, fontweight='bold')
        ax1.set_ylabel('Final SoC [%]', fontsize=11)
        ax1.set_title('Final SoC Stability (Last 500 eps)', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

        # 4-2. 방치 시간 안정성 플롯 개편
        sns.boxplot(
            data=[q_idl[-eval_window:], p_idl[-eval_window:]], 
            palette=[colors['Q-Learning'], colors['Predictive ML']],
            width=0.5, fliersize=4, linewidth=1.2, ax=ax2
        )
        ax2.set_xticklabels(['Q-Learning', 'Predictive ML'], fontsize=10, fontweight='bold')
        ax2.set_ylabel('Overcharge Idle Time [min]', fontsize=11)
        ax2.set_title('Aging Protection Stability (Last 500 eps)', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"4-{run}_Boxplot_Conf.png"), dpi=300)
        plt.close()

    print(f"\n[성공] 학회 발표 논문용 다변화 그래프 파일 출력이 전원 완료되었습니다. '{save_dir}' 내역을 확인하십시오.")
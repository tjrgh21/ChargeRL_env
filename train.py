import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from env import BatteryEnv
from agent import QAgent, quantize_state


def get_next_save_dir(base_name="sim_results"):
    if not os.path.exists(base_name): return base_name
    i = 1
    while os.path.exists(f"{base_name}_{i}"): i += 1
    return f"{base_name}_{i}"



def moving_average(data, window=200):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# --- 비교 정책군 정의 ---
# def run_fixed_policy(episodes=4000):
#     """Naive Fixed: 2시간만 대기 후 바로 충전 (무조건 100% 도달하지만 방치 시간 매우 긺)"""
#     env = BatteryEnv(); rewards, idles, final_socs = [], [], []
#     for _ in range(episodes):
#         env.reset(); done = False; ep_reward = 0
#         while not done:
#             action = 0 if env.current_time < 120 else 1
#             _, reward, done, info = env.step(action)
#             ep_reward += reward
#         rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
#     return rewards, idles, final_socs

def run_predictive_policy(episodes=4000): # 4000 에피소드로 테스트 추천
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    avg_unplug = 450.0 
    # safety_margin = 60.0
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
    NUM_RUNS = 5
    save_dir = get_next_save_dir()
    os.makedirs(save_dir, exist_ok=True)

    print(f"총 {NUM_RUNS}회의 학습을 시작합니다. 결과 저장 폴더: {save_dir}")

    for run in range(1, NUM_RUNS + 1):
        print(f"\n[{run}/{NUM_RUNS}] Run 시작...")
        env = BatteryEnv(); agent = QAgent()
        
        # 각 정책 실행 데이터 수집
        q_rew, q_idl, q_soc = train_qlearning_full(env, agent, EPISODES)
        # f_rew, f_idl, f_soc = run_fixed_policy(EPISODES)
        p_rew, p_idl, p_soc = run_predictive_policy(EPISODES) 
        g_rew, g_idl, g_soc = run_greedy_policy(EPISODES)

        eval_window = 500 # 평가용 최신 에피소드 기준

        # --- 1. Reward Comparison 그래프 저장 ---
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(q_rew), label="Q-Learning", color='red')
        plt.plot(moving_average(p_rew), label="Predictive ML", color='purple', alpha=0.8)
        # plt.plot(moving_average(f_rew), label="Fixed (Naive)", color='blue', alpha=0.5)
        plt.plot(moving_average(g_rew), label="Greedy", color='green', linestyle=':', alpha=0.5)
        
        plt.title(f"Learning Curve: Reward Convergence (Run {run})")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Reward (w=300)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"1-{run}_Reward.png"))
        plt.close()

        # --- 2. Aging Comparison 그래프 저장 ---
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(q_idl), label="Q-Learning", color='red')
        plt.plot(moving_average(p_idl), label="Predictive ML", color='purple', alpha=0.8)
        # plt.plot(moving_average(f_idl), label="Fixed (Naive)", color='blue', alpha=0.5)
        plt.plot(moving_average(g_idl), label="Greedy", color='green', linestyle=':', alpha=0.5)
        
        plt.title(f"Battery Protection: Overcharge Time Reduction (Run {run})")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Idle Time [min] (w=300)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"2-{run}_Aging.png"))
        plt.close()

        # --- 3. Trade-off Scatter Plot (파레토 프론트) ---
        plt.figure(figsize=(10, 8))
        policies = {
            "Q-Learning": (np.mean(q_idl[-eval_window:]), np.mean(q_soc[-eval_window:]), 'red', 'X', 200),
            "Predictive ML": (np.mean(p_idl[-eval_window:]), np.mean(p_soc[-eval_window:]), 'purple', 's', 150),
            # "Fixed (Naive)": (np.mean(f_idl[-eval_window:]), np.mean(f_soc[-eval_window:]), 'blue', 'o', 100),
            "Greedy": (np.mean(g_idl[-eval_window:]), np.mean(g_soc[-eval_window:]), 'green', '^', 100)
        }

        for name, (idle_avg, soc_avg, color, marker, size) in policies.items():
            plt.scatter(idle_avg, soc_avg, color=color, marker=marker, s=size, label=name, edgecolors='black', zorder=5)
            plt.annotate(name, (idle_avg, soc_avg), xytext=(12, 0), textcoords='offset points', 
                         fontsize=11, color=color, fontweight='bold', verticalalignment='center')

        plt.axhline(100.0, color='black', linestyle='--', alpha=0.5, label='Target SoC (100%)')
        plt.title(f"Performance Trade-off Analysis (Run {run})", fontsize=15, fontweight='bold')
        
        plt.xlabel("Average Overcharge Idle Time [min] (Lower is Better ←)", fontsize=13)
        plt.ylabel("Average Final SoC [%] (Higher is Better ↑)", fontsize=13)
        
        # plt.scatter(0, 100, color='gold', marker='*', s=400, edgecolors='black', zorder=10)
        # plt.annotate('Ideal Spot', (0, 100), xytext=(15, -15), textcoords='offset points', fontsize=12, color='darkgoldenrod', fontweight='bold')

        # plt.xlim(max(max(g_idl[-eval_window:]), max(f_idl[-eval_window:]), 350), -10)
        # Fixed가 빠졌으므로 xlim 계산 시 Fixed 제외
        max_idle_time = max(max(g_idl[-eval_window:]), max(p_idl[-eval_window:]))
        plt.xlim(max(max_idle_time, 350), -10) 
        plt.ylim(70, 105) 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"3-{run}_Tradeoff_Scatter.png"), dpi=300)
        plt.close()

        # --- 4. 성능 안정성 박스플롯 (Robustness Box Plot) ---
        plt.figure(figsize=(12, 5))
        
        # 4-1. SoC 안정성
        plt.subplot(1, 2, 1)
        sns.boxplot(data=[q_soc[-eval_window:], p_soc[-eval_window:]], palette=['red', 'purple'])
        plt.xticks([0, 1], ['Q-Learning', 'Predictive ML'], fontsize=11)
        plt.ylabel('Final SoC [%]', fontsize=12)
        plt.title('Final SoC Stability (Last 500 eps)', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)

        # 4-2. 방치 시간 안정성
        plt.subplot(1, 2, 2)
        sns.boxplot(data=[q_idl[-eval_window:], p_idl[-eval_window:]], palette=['red', 'purple'])
        plt.xticks([0, 1], ['Q-Learning', 'Predictive ML'], fontsize=11)
        plt.ylabel('Overcharge Idle Time [min]', fontsize=12)
        plt.title('Aging Protection Stability (Last 500 eps)', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"4-{run}_Boxplot.png"), dpi=300)
        plt.close()

    print(f"\n모든 시뮬레이션 완료. '{save_dir}' 폴더를 확인하세요.")
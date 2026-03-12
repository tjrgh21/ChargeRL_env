import numpy as np
import matplotlib.pyplot as plt
import os
import random
from env import BatteryEnv
from agent import QAgent, quantize_state

def get_next_save_dir(base_name="sim_results"):
    if not os.path.exists(base_name): return base_name
    i = 1
    while os.path.exists(f"{base_name}_{i}"): i += 1
    return f"{base_name}_{i}"

def moving_average(data, window=500):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

# --- 비교 정책군 정의 ---
def run_fixed_policy(episodes=3000):
    """Naive Fixed: 1시간만 대기 후 바로 충전"""
    env = BatteryEnv(); rewards, idles = [], []
    for _ in range(episodes):
        env.reset(); done = False; ep_reward = 0
        while not done:
            action = 0 if env.current_time < 120 else 1 
            _, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward); idles.append(info["idle_time"])
    return rewards, idles

def run_random_policy(episodes=3000):
    """[추가] Random Policy: 매 순간 무작위로 충전/대기 결정"""
    env = BatteryEnv(); rewards, idles = [], []
    for _ in range(episodes):
        env.reset(); done = False; ep_reward = 0
        while not done:
            action = random.randint(0, 1) # 0 또는 1 무작위 선택
            _, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward); idles.append(info["idle_time"])
    return rewards, idles

def run_greedy_policy(episodes=3000):
    """Greedy Policy: 즉시 충전"""
    env = BatteryEnv(); rewards, idles = [], []
    for _ in range(episodes):
        env.reset(); done = False; ep_reward = 0
        while not done:
            action = 1 
            _, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward); idles.append(info["idle_time"])
    return rewards, idles

def train_qlearning_full(env, agent, episodes=3000):
    """Q-Learning 학습 루프"""
    rewards, idles, success_history = [], [], []
    print(f"{'Episode':<10} | {'Total Reward':<15} | {'Final SoC':<10} | {'Idle Time':<10}")
    print("-" * 55)
    
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
        rewards.append(ep_reward); idles.append(info["idle_time"])
        success_history.append(1 if info["soc"] >= 95 else 0)
        
        if (ep + 1) % 100 == 0:
            print(f"{ep+1:<10} | {ep_reward:<15.2f} | {info['soc']:<10.1f}% | {info['idle_time']:<10} min")

    return rewards, idles, success_history

if __name__ == "__main__":
    EPISODES = 3000
    save_dir = get_next_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    env = BatteryEnv(); agent = QAgent()
    
    print("Q-Learning 학습 시작...")
    q_rew, q_idl, q_acc = train_qlearning_full(env, agent, EPISODES)
    
    print("\n대조군(Fixed, Random, Greedy) 시뮬레이션 중...")
    f_rew, f_idl = run_fixed_policy(EPISODES)
    r_rew, r_idl = run_random_policy(EPISODES) # Random 실행 추가
    g_rew, g_idl = run_greedy_policy(EPISODES)

    # 1. Reward Comparison 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(q_rew), label="Q-Learning (Proposed AI)", color='red', linewidth=2.5)
    plt.plot(moving_average(f_rew), label="Fixed Policy", color='blue', alpha=0.7)
    plt.plot(moving_average(r_rew), label="Random Policy", color='gray', linestyle='--') # Random 추가
    plt.plot(moving_average(g_rew), label="Greedy Policy (Immediate)", color='green', alpha=0.5)
    plt.title("Total Reward Comparison"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "1_Reward.png"))

    # 2. Aging Comparison 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(q_idl), label="Q-Learning (Proposed AI)", color='red', linewidth=2.5)
    plt.plot(moving_average(f_idl), label="Fixed Policy", color='blue', alpha=0.7)
    plt.plot(moving_average(r_idl), label="Random Policy", color='gray', linestyle='--') # Random 추가
    plt.plot(moving_average(g_idl), label="Greedy Policy", color='green', alpha=0.5)
    plt.title("Battery Aging Protection (Lower is Better)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "2_Aging.png"))
    
    print(f"\n모든 시뮬레이션 완료. 결과 저장 폴더: {save_dir}")
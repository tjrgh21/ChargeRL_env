import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from env_2case import BatteryEnv  # 새로 생성한 파일명으로 임포트 변경
from agent import QAgent, quantize_state  # agent.py는 기존 파일 유지

def get_next_save_dir(base_name="sim_results_2case"):
    if not os.path.exists(base_name): return base_name
    i = 1
    while os.path.exists(f"{base_name}_{i}"): i += 1
    return f"{base_name}_{i}"

# 변화를 예민하게 보기 위해 윈도우 사이즈를 100으로 줄임
def moving_average(data, window=100):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def run_predictive_policy(episodes=6000, shift_ep=4000): 
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    avg_unplug = 450.0 
    safety_margin = 70.0
    
    for ep in range(episodes):
        if ep == shift_ep:
            env.set_phase(2) # 4000 에피소드에서 환경 급변
            
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

def run_greedy_policy(episodes=6000, shift_ep=4000):
    env = BatteryEnv(); rewards, idles, final_socs = [], [], []
    for ep in range(episodes):
        if ep == shift_ep:
            env.set_phase(2)
            
        env.reset(); done = False; ep_reward = 0
        while not done:
            action = 1
            _, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward); idles.append(info["idle_time"]); final_socs.append(info["soc"])
    return rewards, idles, final_socs

def train_qlearning_full(env, agent, episodes=6000, shift_ep=4000):
    rewards, idles, final_socs = [], [], []
    
    # 물리적 지표 단기 기억 버퍼
    recent_idles = []
    recent_socs = []
    cooldown = 0 
    
    print(f"{'Episode':<10} | {'Total Reward':<15} | {'Final SoC':<10} | {'Epsilon'}")
    for ep in range(episodes):
        
        # 시스템 개입: 환경만 조용히 바꿈
        if ep == shift_ep:
            env.set_phase(2) 
            
        if cooldown > 0:
            cooldown -= 1
            
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            state_disc = quantize_state(state)
            action = agent.choose_action(state_disc)
            next_state, reward, done, info = env.step(action)
            agent.learn(state_disc, action, reward, quantize_state(next_state), ep)
            state = next_state
            ep_reward += reward
        
        # 물리적 결과 수집
        idle_time = info["idle_time"]
        final_soc = info["soc"]
        
        # ----------------------------------------------------------------
        # 🤖 [에이전트 자율 적응 로직 V3 - 물리 지표 기반]
        # ----------------------------------------------------------------
        recent_idles.append(idle_time)
        recent_socs.append(final_soc)
        
        if len(recent_idles) > 50:
            recent_idles.pop(0)
            recent_socs.pop(0)
            
            avg_idle = np.mean(recent_idles)
            avg_soc = np.mean(recent_socs)
            
            # 확신 상태(eps < 0.1) & 쿨다운 종료 상태에서 지표 검사
            if agent.epsilon < 0.1 and cooldown == 0:
                # 조건 1: 최근 50번 평균 방치 시간이 120분을 넘어섬 (지연 기상 패턴으로 급변)
                # 조건 2: 최근 50번 평균 충전율이 90% 밑으로 떨어짐 (조기 기상 패턴으로 급변)
                if avg_idle > 120.0 or avg_soc < 90.0:
                    print("-" * 65)
                    print(f"🚨 [Ep {ep}] 에이전트 자율 감지: 물리적 패턴 급변 인지!")
                    print(f"   -> 원인: 평균 방치시간 {avg_idle:.1f}분 / 평균 SoC {avg_soc:.1f}%")
                    print(f"   -> 대처: 탐험(Epsilon)을 0.5로 강제 부스팅하여 재수렴을 시작합니다.")
                    print("-" * 65)
                    
                    agent.epsilon = 0.5   
                    cooldown = 400        # 400 에피소드 동안 재학습 대기
                    recent_idles = []     # 버퍼 리셋
                    recent_socs = []
        # ----------------------------------------------------------------
        
        agent.decay_epsilon()
        rewards.append(ep_reward)
        idles.append(idle_time)
        final_socs.append(final_soc)
        
        if (ep + 1) % 500 == 0:
            print(f"{ep+1:<10} | {ep_reward:<15.2f} | {info['soc']:<10.1f}% | eps: {agent.epsilon:.2f}")
            
    return rewards, idles, final_socs

if __name__ == "__main__":
    EPISODES = 6000     # 재수렴 관찰을 위해 6000으로 확장
    SHIFT_EP = 4000     # 환경 급변 트리거 시점
    NUM_RUNS = 10
    save_dir = get_next_save_dir()
    os.makedirs(save_dir, exist_ok=True)

    print(f"총 {NUM_RUNS}회의 학습을 시작합니다. 결과 저장 폴더: {save_dir}")

    for run in range(1, NUM_RUNS + 1):
        print(f"\n[{run}/{NUM_RUNS}] Run 시작...")
        env = BatteryEnv(); agent = QAgent()
        
        q_rew, q_idl, q_soc = train_qlearning_full(env, agent, EPISODES, SHIFT_EP)
        p_rew, p_idl, p_soc = run_predictive_policy(EPISODES, SHIFT_EP) 
        g_rew, g_idl, g_soc = run_greedy_policy(EPISODES, SHIFT_EP)

        eval_window = 500 # 최종 수렴 여부는 마지막 500에피소드 기준으로 평가

        # --- 1. Reward Comparison 그래프 저장 ---
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(q_rew), label="Q-Learning", color='red')
        plt.plot(moving_average(p_rew), label="EMA Model", color='purple', alpha=0.8)
        plt.plot(moving_average(g_rew), label="Greedy", color='green', linestyle=':', alpha=0.5)
        
        # 환경 변화 시점 수직선 추가
        plt.axvline(x=SHIFT_EP, color='black', linestyle='--', linewidth=1.5, label='Environment Shift')
        
        plt.title(f"Learning Curve: Reward Convergence & Robustness")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Reward (w=100)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"1-{run}_Reward.png"))
        plt.close()

        # --- 2. Aging Comparison 그래프 저장 ---
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(q_idl), label="Q-Learning", color='red')
        plt.plot(moving_average(p_idl), label="EMA Model", color='purple', alpha=0.8)
        plt.plot(moving_average(g_idl), label="Greedy", color='green', linestyle=':', alpha=0.5)
        
        # 환경 변화 시점 수직선 추가
        plt.axvline(x=SHIFT_EP, color='black', linestyle='--', linewidth=1.5, label='Environment Shift')
        
        plt.title(f"Battery Protection: Overcharge Time vs Environment Shift")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Idle Time [min] (w=100)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"2-{run}_Aging.png"))
        plt.close()

        # --- 3. Trade-off Scatter Plot (파레토 프론트) ---
        plt.figure(figsize=(10, 8))
        policies = {
            "Q-Learning": (np.mean(q_idl[-eval_window:]), np.mean(q_soc[-eval_window:]), 'red', 'X', 200),
            "EMA Model": (np.mean(p_idl[-eval_window:]), np.mean(p_soc[-eval_window:]), 'purple', 's', 150),
            "Greedy": (np.mean(g_idl[-eval_window:]), np.mean(g_soc[-eval_window:]), 'green', '^', 100)
        }

        for name, (idle_avg, soc_avg, color, marker, size) in policies.items():
            plt.scatter(idle_avg, soc_avg, color=color, marker=marker, s=size, label=name, edgecolors='black', zorder=5)
            plt.annotate(name, (idle_avg, soc_avg), xytext=(12, 0), textcoords='offset points', 
                         fontsize=11, color=color, fontweight='bold', verticalalignment='center')

        plt.axhline(100.0, color='black', linestyle='--', alpha=0.5, label='Target SoC (100%)')
        plt.title(f"Performance Trade-off Analysis (Phase 2 Converged)", fontsize=15, fontweight='bold')
        
        plt.xlabel("Average Overcharge Idle Time [min] (Lower is Better ←)", fontsize=13)
        plt.ylabel("Average Final SoC [%] (Higher is Better ↑)", fontsize=13)
        
        max_idle_time = max(max(g_idl[-eval_window:]), max(p_idl[-eval_window:]))
        plt.xlim(max(max_idle_time, 350), -10) 
        plt.ylim(70, 105) 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"3-{run}_Tradeoff_Scatter.png"), dpi=300)
        plt.close()

        # --- 4. 성능 안정성 박스플롯 (Robustness Box Plot) ---
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=[q_soc[-eval_window:], p_soc[-eval_window:]], palette=['red', 'purple'])
        plt.xticks([0, 1], ['Q-Learning', 'EMA Model'], fontsize=11)
        plt.ylabel('Final SoC [%]', fontsize=12)
        plt.title('Final SoC Stability (Post-Shift Converged)', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)

        plt.subplot(1, 2, 2)
        sns.boxplot(data=[q_idl[-eval_window:], p_idl[-eval_window:]], palette=['red', 'purple'])
        plt.xticks([0, 1], ['Q-Learning', 'EMA Model'], fontsize=11)
        plt.ylabel('Overcharge Idle Time [min]', fontsize=12)
        plt.title('Aging Protection Stability (Post-Shift Converged)', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"4-{run}_Boxplot.png"), dpi=300)
        plt.close()

    print(f"\n모든 시뮬레이션 완료. '{save_dir}' 폴더를 확인하세요.")
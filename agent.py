import numpy as np



def quantize_state(state):
    day, current_time_min, soc = state

    # 15분 단위로 시간이 정확히 한 칸씩 전진합니다. (타임루프 완벽 차단)
    time_step = min(95, int(current_time_min // 15))

    # [핵심 1] 19 -> 20으로 변경하여 총 21칸(0~20) 생성!
    # 이제 99.9% 이하는 19번 방, 완벽한 100.0%는 20번 방으로 명확히 구분됩니다.
    soc_step = min(20, int(soc // 5))

    return (int(day), time_step, soc_step)



class QAgent:
    # 10 -> 20
    def __init__(self, state_shape=(7, 96, 21), n_actions=2, alpha=0.1, gamma=0.99):
        self.Q_table = np.zeros(state_shape + (n_actions,))
        self.alpha_initial = 0.1  # 초기 학습률 저장
        self.alpha = alpha     # 학습률
        self.gamma = gamma     # 할인율 (미래 보상 가치 중시)
        self.epsilon = 1.0     # 초기 탐험율
        self.epsilon_decay = 0.995  # 탐험율을 조금 더 천천히 감소시켜 충분한 탐색 유도
        self.epsilon_min = 0.001    # 최소 탐험율을 낮춰 수렴 시 진동 억제



    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.Q_table[state])


    def learn(self, state, action, reward, next_state, episode):

        self.alpha = self.alpha_initial / (1 + 0.001 * episode)


        best_next = np.max(self.Q_table[next_state])
        current_q = self.Q_table[state + (action,)]


        # 벨만 방정식을 이용한 Q-값 업데이트
        self.Q_table[state + (action,)] += self.alpha * (
            reward + self.gamma * best_next - current_q
        )



    def decay_epsilon(self):
        # 학습이 진행됨에 따라 점진적으로 탐험 비중을 낮춤
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
import numpy as np



def quantize_state(state):

    day, hour, soc, pattern = state

    time_step = min(23, int(hour))

    soc_step = min(9, int(soc // 10))

    return (int(day), time_step, soc_step, int(pattern))



class QAgent:

    def __init__(self, state_shape=(7, 24, 10, 3), n_actions=2, alpha=0.1, gamma=0.99):

        self.Q_table = np.zeros(state_shape + (n_actions,))

        self.alpha_initial = 0.1  # 초기 학습률 저장

        self.alpha = self.alpha_initial     # 학습률

        self.gamma = gamma     # 할인율 (미래 보상 가치 중시)

        self.epsilon = 1.0     # 초기 탐험율

        self.epsilon_decay = 0.998  # 탐험율을 조금 더 천천히 감소시켜 충분한 탐색 유도

        self.epsilon_min = 0.0    # 최소 탐험율을 낮춰 수렴 시 진동 억제



    def choose_action(self, state):

        if np.random.rand() < self.epsilon:

            return np.random.randint(2)

        return np.argmax(self.Q_table[state])



    def learn(self, state, action, reward, next_state, episode):

        # 학습 중반부 이후부터는 변화를 작게 주어 안정적으로 수렴 유도

        self.alpha = self.alpha_initial / (1 + 0.005 * episode)



        best_next = np.max(self.Q_table[next_state])

        current_q = self.Q_table[state + (action,)]



        # 벨만 방정식을 이용한 Q-값 업데이트

        self.Q_table[state + (action,)] += self.alpha * (

            reward + self.gamma * best_next - current_q

        )



    def decay_epsilon(self):

        # 학습이 진행됨에 따라 점진적으로 탐험 비중을 낮춤

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
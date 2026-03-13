import numpy as np
import random



class BatteryEnv:

    def __init__(self):
        self.reset()



    def reset(self):

        """[환경 초기화] 사용자가 충전기를 꽂는 순간부터 시작"""

        self.day = random.randint(0, 6)
        self.current_time = 0 # 충전 시작 후 경과 시간 (분)
        self.soc = random.uniform(0, 100) # 초기 SoC (0~100)
        self.pattern = random.randint(0, 2)

       

        # 기기 해제 시각 (정답 데이터)

        if self.day < 5:
            self.actual_unplug_time = 420 + random.normalvariate(0, 30)
        else:
            self.actual_unplug_time = 600 + random.normalvariate(0, 60)

       

        # [추가] 물리적 최대 가능 SoC 계산 (연결 즉시 풀가동 시 도달 가능한 SoC)

        # 평균 충전율을 분당 약 1.2%로 가정하여 계산
        total_duration = self.actual_unplug_time
        self.potential_soc = min(100.0, self.soc + (1.2 * total_duration))
       

        self.is_charging = False
        self.full_charge_time = 0 # 완충 시점 기록용

        return self._get_state()



    def _get_state(self):

        # 경과 시간을 시간 단위로 변환하여 에이전트에게 전달
        return np.array([self.day, self.current_time // 60, self.soc, self.pattern])



    def step(self, action):

        """action 0: 대기 (Wait), action 1: 충전 (Charge)"""
        # 1. 액션 수행 및 SoC 업데이트

        if action == 1:

            self.is_charging = True
            charge_rate = 1.4 if self.soc < 80 else 0.6
            self.soc = min(100.0, self.soc + charge_rate)

            # 처음 100%에 도달한 시점 기록

            if self.soc >= 100.0 and self.full_charge_time == 0:
                self.full_charge_time = self.current_time

        else:
            self.is_charging = False



        # 2. 시간 흐름

        self.current_time += 1
        done = self.current_time >= self.actual_unplug_time

       

        # 3. 보상 및 지표 계산

        reward = 0.0
        idle_time = 0

       

        if not done:

            # 매 스텝마다 주는 소량의 '배터리 보호 보상'
            if not self.is_charging and self.soc < 100:
                reward = 0.05 # e대기 보상 유지

        else:

            # 최종 기기 해제 시점의 정산
            if self.full_charge_time > 0:
                idle_time = self.actual_unplug_time - self.full_charge_time

           

            # [수정] UX 중심의 공정한 보상 로직
            if self.soc >= 99.5: # 사실상 완충 성공
                # 완충 보상 100점에서 방치 시간 감점 적용
                reward = 150.0 - (idle_time * 0.5)

            else:
                # 완충 실패 시 상황 분석
                if self.potential_soc < 99.5:

                    # 물리적 한계 상황에서는 감점 최소화
                    reward = 50.0 - (self.potential_soc - self.soc) * 1.0

                else:

                    # 에이전트 과실 시에만 음수 보상 부여
                    reward = -(100.0 - self.soc) * 1.0

        return self._get_state(), reward, done, {"soc": self.soc, "idle_time": idle_time}
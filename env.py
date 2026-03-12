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
        total_duration = self.actual_unplug_time
        # 평균 충전율을 분당 약 1.2%로 가정하여 계산
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
            # 배터리 구간별 충전 속도 차별화 (물리적 특성 반영)
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
            # [수정] 매 스텝 주던 미세 보상을 0으로 설정하여 수렴 후 그래프 진동 억제
            reward = 0.0
        else:
            # 최종 기기 해제 시점의 정산
            if self.full_charge_time > 0:
                idle_time = self.actual_unplug_time - self.full_charge_time
            
            # (1) UX 성공: 99.5% 이상 완충
            if self.soc >= 99.5:
                # 방치 시간(idle_time)에 비례한 감점 적용 (최고 100점)
                reward = 100.0 - (idle_time * 0.5) 
            
            # (2) 미완충 실패
            else:
                # (A) 물리적 한계: 즉시 충전했어도 99.5%가 불가능했던 경우
                if self.potential_soc < 99.5:
                    # 부족한 양만큼만 가볍게 패널티 (에이전트 책임 아님)
                    gap = max(0, self.potential_soc - self.soc)
                    reward = -gap 
                # (B) 에이전트 과실: 시간은 충분했는데 대기를 너무 오래 함
                else:
                    # UX 저해에 대한 강력한 패널티 부여 (시작점을 낮게 만드는 요인)
                    reward = -150.0 

        return self._get_state(), reward, done, {"soc": self.soc, "idle_time": idle_time}
import numpy as np
import random

class BatteryEnv:
    def __init__(self):
        self.phase = 1  # 1: 기존 환경, 2: 급변된 환경
        self.reset()

    def set_phase(self, phase):
        """외부(train_2case.py)에서 환경 변화를 트리거하기 위한 메서드"""
        self.phase = phase

    def reset(self):
        """[환경 초기화] 사용자가 충전기를 꽂는 순간부터 시작"""
        self.day = random.randint(0, 6)
        self.current_time = 0 # 충전 시작 후 경과 시간 (분)
        self.soc = random.uniform(0, 100) # 초기 SoC (0~100)

        # Phase에 따른 기상 패턴 완전 분기
        if self.phase == 1:
            # [Phase 1] 기존 7일 기상 패턴
            if self.day == 0:   self.actual_unplug_time = 420 + random.normalvariate(0, 40)
            elif self.day == 1: self.actual_unplug_time = 330 + random.normalvariate(0, 40)
            elif self.day == 2: self.actual_unplug_time = 510 + random.normalvariate(0, 45)
            elif self.day == 3: self.actual_unplug_time = 360 + random.normalvariate(0, 35)
            elif self.day == 4: self.actual_unplug_time = 450 + random.normalvariate(0, 40)
            elif self.day == 5: self.actual_unplug_time = 660 + random.normalvariate(0, 80)
            elif self.day == 6: self.actual_unplug_time = 540 + random.normalvariate(0, 90)
        
        elif self.phase == 2:
            # [Phase 2] 환경 급변: 라이프스타일 변화로 모든 기상 시간이 약 3시간(180분) 지연됨
            if self.day == 0:   self.actual_unplug_time = 600 + random.normalvariate(0, 50)
            elif self.day == 1: self.actual_unplug_time = 510 + random.normalvariate(0, 50)
            elif self.day == 2: self.actual_unplug_time = 690 + random.normalvariate(0, 60)
            elif self.day == 3: self.actual_unplug_time = 540 + random.normalvariate(0, 50)
            elif self.day == 4: self.actual_unplug_time = 630 + random.normalvariate(0, 50)
            elif self.day == 5: self.actual_unplug_time = 840 + random.normalvariate(0, 90)
            elif self.day == 6: self.actual_unplug_time = 720 + random.normalvariate(0, 100)

        # 평균 충전율을 분당 약 1.2%로 가정하여 계산
        total_duration = self.actual_unplug_time
        self.potential_soc = min(100.0, self.soc + (1.2 * total_duration))

        self.is_charging = False
        self.full_charge_time = 0 # 완충 시점 기록용

        return self._get_state()

    def _get_state(self):
        return np.array([self.day, self.current_time, self.soc])
    
    def step(self, action):
        step_size = 15  # 환경을 15분 단위로 진행
        
        if action == 1:
            self.is_charging = True
            for i in range(step_size):
                if self.soc < 100.0:
                    charge_rate = 1.4 if self.soc < 80 else 0.6
                    self.soc = min(100.0, self.soc + charge_rate)
                    if self.soc >= 100.0 and self.full_charge_time == 0:
                        self.full_charge_time = self.current_time + i
        else:
            self.is_charging = False

        self.current_time += step_size
        done = self.current_time >= self.actual_unplug_time

        reward = 0.0
        idle_time = 0

        if done and self.full_charge_time > 0:
            idle_time = self.actual_unplug_time - self.full_charge_time

        if not done:
            if self.soc >= 100.0:
                # 30분 UX 유예시간 (Grace Period)
                overcharge_time = self.current_time - self.full_charge_time
                if overcharge_time <= 30:
                    reward = 0.0
                else:
                    reward = -2.0 * step_size
            elif self.soc < 80.0 and not self.is_charging:
                reward = 0.05 * step_size
            elif not self.is_charging:
                reward = 0.0
            elif self.is_charging:
                reward = 0.0

        else:
            if self.soc >= 99.5: 
                reward = 400.0
            else:
                if self.potential_soc < 99.5:
                    reward = 50.0 - (self.potential_soc - self.soc) * 1.0
                else:
                    reward = -(100.0 - self.soc) * 20.0

        return self._get_state(), reward, done, {"soc": self.soc, "idle_time": idle_time}
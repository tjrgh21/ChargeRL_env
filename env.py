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
        # self.pattern = random.randint(0, 2)

        # 7일 내내 완전히 다른 기상 패턴
        if self.day == 0:   
            self.actual_unplug_time = 420 + random.normalvariate(0, 40)
        elif self.day == 1: 
            self.actual_unplug_time = 330 + random.normalvariate(0, 40)
        elif self.day == 2: 
            self.actual_unplug_time = 510 + random.normalvariate(0, 45)
        elif self.day == 3: 
            self.actual_unplug_time = 360 + random.normalvariate(0, 35)
        elif self.day == 4: 
            self.actual_unplug_time = 450 + random.normalvariate(0, 40)
        elif self.day == 5: 
            self.actual_unplug_time = 660 + random.normalvariate(0, 80)
        elif self.day == 6: 
            self.actual_unplug_time = 540 + random.normalvariate(0, 90)
       
        # 평균 충전율을 분당 약 1.2%로 가정하여 계산
        total_duration = self.actual_unplug_time
        self.potential_soc = min(100.0, self.soc + (1.2 * total_duration))

        self.is_charging = False
        self.full_charge_time = 0 # 완충 시점 기록용

        return self._get_state()


    def _get_state(self):

        return np.array([self.day, self.current_time, self.soc])
    
    def step(self, action):
        step_size = 15  # [핵심] 환경을 1분 대신 15분 단위로 큼직하게 진행!
        
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
                # [사용자 아이디어 적용] 30분 UX 유예시간 (Grace Period)
                overcharge_time = self.current_time - self.full_charge_time

                if overcharge_time <= 30:
                    # 100% 도달 후 30분 이내: 현실적인 방치 허용 구간 (페널티 없음)
                    reward = 0.0
                else:
                    # 30분 초과 시: 배터리 수명에 악영향을 주므로 가차 없이 페널티 부여
                    reward = -2.0 * step_size
            # elif not self.is_charging:
            #     #대기 보상
            #     reward = 0.0

                # 과충전 페널티 (-15점/15분)
                # reward = -1.0 * step_size
            elif self.soc < 80.0 and not self.is_charging:
                # 0.5는 너무 큽니다! 0.1 이하로 유지하여 
                # '버티는 게 아주 살짝 좋긴 한데, 완충 못하면 끝장이다'라는 걸 인지시켜야 합니다.
                reward = 0.05 * step_size

            elif not self.is_charging:
                #대기 보상
                reward = 0.0
            elif self.is_charging:
                # 충전 중일 때는 기본 0점 (목표를 향해 가는 중)
                reward = 0.0

        else:
            if self.soc >= 99.5: 
                reward = 300.0
            else:
                if self.potential_soc < 99.5:
                    reward = 50.0 - (self.potential_soc - self.soc) * 1.0
                else:
                    reward = -(100.0 - self.soc) * 20.0

        return self._get_state(), reward, done, {"soc": self.soc, "idle_time": idle_time}

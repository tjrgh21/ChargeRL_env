import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats

# 폰트 및 스타일 설정
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# X축 구간 설정
x_minutes = np.linspace(180, 960, 1000)

# 통계학(Law of Total Variance)을 통해 도출된 통합 파라미터 (내부 계산용으로만 사용)
mu_wd, sigma_wd = 414, 76    # 통합 평일
mu_we, sigma_we = 600, 104   # 통합 주말

fig, ax = plt.subplots(figsize=(10, 5))

# 평일 통합 그래프 (파란색)
y_wd = stats.norm.pdf(x_minutes, mu_wd, sigma_wd)
ax.plot(x_minutes, y_wd, color='#1f77b4', linewidth=3.0, label='Weekday') # 수치 제거
ax.fill_between(x_minutes, y_wd, color='#1f77b4', alpha=0.2)

# 주말 통합 그래프 (주황/빨간색)
y_we = stats.norm.pdf(x_minutes, mu_we, sigma_we)
ax.plot(x_minutes, y_we, color='#d62728', linewidth=3.0, label='Weekend') # 수치 제거
ax.fill_between(x_minutes, y_we, color='#d62728', alpha=0.2)

# X축 분(Minutes) 단위를 'HH:MM' 포맷으로 변환
def format_func(value, tick_number):
    hour = int(value // 60)
    minute = int(value % 60)
    return f"{hour:02d}:{minute:02d}"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
ax.set_xticks(np.arange(180, 961, 60)) # 1시간 간격

# 그래프 꾸미기
ax.set_title('Aggregated Actual Environment: Wake-up Distribution', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Time of Day', fontsize=13, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)

# 범례 표시 (폰트 크기를 약간 키워 가독성 확보)
ax.legend(fontsize=14, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('combined_distribution_ppt_clean.png', dpi=300, transparent=False)
plt.show()
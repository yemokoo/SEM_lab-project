import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# 데이터 준비
data = [
    234963, 235461, 236078, 236716, 237161, 237565, 238107, 238401, 238801, 239399, 239909, 240414,
    241180, 241541, 242112, 242582, 242905, 243492, 243878, 244276, 244840, 245433, 245849, 246163,
    246900, 247459, 248173, 248860, 249543, 250307, 251121, 251680, 252217, 252959, 253797, 254545,
    255462, 256067, 256996, 257879, 258565, 259372, 260124, 260751, 261418, 261868, 262485, 262945,
    263637, 263983, 264378, 264756, 265083, 265592, 265942, 266258, 266620, 266967, 267334, 267610,
    268074, 268292, 268524, 268658, 268783, 268941, 269026, 269113, 269167, 269051, 268984, 269099,
    269500, 269849, 270072, 270335, 270608, 271118, 271629, 271981, 272534, 273065, 273582, 274079,
    274766, 275315, 275854, 276082, 276346, 276896, 277405, 277786, 278159, 278537, 279019, 279304,
    280015, 280439, 280822, 280946, 281079, 281526, 282279, 282837, 283402, 283960, 284607, 285066,
    285395, 285602, 285738, 285840, 286133, 286398, 286818, 287219, 287654, 287926, 288310, 288723,
    289463, 289890, 290434, 290659, 290861, 291270, 291767, 292085, 292437, 292788, 293069, 293513
]
dates = pd.date_range(start='2014-01-01', periods=len(data), freq='MS')
df = pd.DataFrame({'Date': dates, 'Registration': data})
df = df.set_index('Date')

# Auto ARIMA 모델 학습
auto_arima_model = pm.auto_arima(df['Registration'],
                                 start_p=0, start_q=0,
                                 test='adf',
                                 max_p=3, max_q=3,
                                 m=1,
                                 d=None,
                                 seasonal=False,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

# 최적 모델 요약 출력
print(auto_arima_model.summary())

# 2025년부터 2030년까지 예측
future_dates = pd.date_range(start='2025-01-01', end='2030-12-01', freq='MS')
forecast_arima = auto_arima_model.predict(n_periods=len(future_dates))
forecast_arima_series = pd.Series(forecast_arima, index=future_dates)

# 전동화율 및 실효 주행 여정 비율
electrification_rate = 0.10
effective_mileage_rate = 0.1877

# 실효 차량 대수 계산
effective_vehicles = forecast_arima_series * electrification_rate * effective_mileage_rate

# 결과 시각화 (실효 차량 대수 그래프 제외)
plt.figure(figsize=(12, 6))
plt.plot(df['Registration'], label='Historical Data')
plt.plot(forecast_arima_series, label='ARIMA Predictions')
plt.xlabel('Date')
plt.ylabel('Number of Vehicles')
plt.title('Car Registration Prediction')
plt.legend()
plt.grid(True)
plt.show()


# 연도별 평균 예측값 및 실효 차량 대수 계산 및 출력
for year in range(2025, 2031):
    year_str = str(year)
    avg_prediction = forecast_arima_series[year_str].mean()
    avg_effective = effective_vehicles[year_str].mean()
    print(f'{year}년 평균 예상 차량 등록 대수: {avg_prediction:.3f}')
    print(f'{year}년 평균 실효 차량 대수: {avg_effective:.3f}\n')
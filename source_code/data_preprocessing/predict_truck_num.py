import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # ARIMA 모델 import

# 데이터 준비 (이전 선형 회귀 코드와 동일)
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

# ARIMA 모델 학습 (order=(p, d, q) 파라미터 설정)
# 여기서는 간단하게 (1, 1, 1) order 사용
model_arima = ARIMA(df['Registration'], order=(1, 1, 1))
model_arima_fit = model_arima.fit() # 모델 학습 완료

# 2030년 예측
future_dates_lr = pd.date_range(start='2025-01-01', end='2030-12-01', freq='MS')
forecast_arima = model_arima_fit.forecast(steps=len(future_dates_lr)) # 미래 예측

# 예측 결과를 pandas Series로 변환 (날짜 인덱스 설정)
forecast_arima_series = pd.Series(forecast_arima, index=future_dates_lr)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Registration'], label='Historical Data')
plt.plot(forecast_arima_series, label='ARIMA Predictions', color='red') # 예측 결과 그래프
plt.xlabel('Date')
plt.ylabel('Registration')
plt.title('Car Registration Prediction (ARIMA)')
plt.legend()
plt.grid(True)
plt.show()

# 2030년 예측값 출력
print("\n2030년 예측값 (ARIMA):")
print(forecast_arima_series['2030']) # 2030년 예측값 출력

# 2030년 평균 예측값
prediction_2030_arima = forecast_arima_series['2030'].mean()
print(f'2030년 예상 차량 등록 대수 (평균, ARIMA): {prediction_2030_arima:.3f}')
import pandas as pd
import matplotlib.pyplot as plt

LOCA_1 = pd.read_csv('DB/12_100010_60.csv')
plt.plot(LOCA_1['ZINST70'])
plt.plot(LOCA_1['QPRZP'])
plt.grid()
plt.show()

#데이터전처리
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()

LOCA_1_VAL_1 = LOCA_1['ZINST70'].to_numpy()
print(np.shape(LOCA_1_VAL_1), type(LOCA_1_VAL_1)) #1번.[1, 2, 3, .....] = (138, ) 모양 , 데이터 138개가 하나의박스에 들어있다

LOCA_1_VAL_1 = LOCA_1_VAL_1.reshape((138,1)) #2번.[[1], [2], [3], ....] = (138,1) 모양으로 바꿔줌, 데이터138개가 각각 1개의 박스에 들어있다.
print(np.shape(LOCA_1_VAL_1), type(LOCA_1_VAL_1))

scaler.fit(LOCA_1_VAL_1) #fit가 원하는 input의 모양은 2번이라서 위에서 모양을 바꿔준것
print(scaler.data_max_)

LOCA_1_VAL_1_OUT = scaler.transform(LOCA_1_VAL_1)
plt.plot(LOCA_1_VAL_1_OUT)
plt.show()

print(LOCA_1.loc[:,['ZINST70','QPRZP']])
LOCA_1_VAL_1_LIST = LOCA_1.loc[:,['ZINST70','QPRZP']].to_numpy()
print(np.shape(LOCA_1_VAL_1_LIST), type(LOCA_1_VAL_1_LIST))

scaler.fit(LOCA_1_VAL_1_LIST)

LOCA_1_VAL_1_OUT = scaler.transform(LOCA_1_VAL_1_LIST)
plt.plot(LOCA_1_VAL_1_OUT)
plt.show()

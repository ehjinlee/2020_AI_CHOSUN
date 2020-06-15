import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#SGTR 전열관 파열
SGTR_1 = pd.read_csv('DB/13_10004_60.csv') #ctrl+spacebar 목록 불러오기

SGTR_1['La'] = 0
print(SGTR_1['La'])

SGTR_1['La'].iloc[0:12] = 1 #사고가 일어난 시점이 12니까 0부터 11까지는 1, 12부터는 0이 되어야 한다

# plt.plot(SGTR_1['ZINST70'])
# plt.plot(SGTR_1['La'])
# plt.show()

train_x = SGTR_1.loc[:,['ZINST70','QPRZP']]
scaler = MinMaxScaler()
scaler.fit(train_x)

train_x = scaler.transform(train_x)
train_y = SGTR_1['La'].to_numpy()
print(np.shape(train_x), type(train_x))
print(np.shape(train_y), type(train_y))

# plt.plot(train_x)
# plt.plot(train_y)
# plt.show()

import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2,
                          activation='sigmoid'),
    tf.keras.layers.Dense(2,
                          activation='softmax'),
])
out = model.predict(train_x[:])
# plt.plot(train_x)
# plt.plot(train_y)
# plt.plot(out)
# plt.show()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=50)
out_trained = model.predict(train_x[:])
#out_max = np.argmax(out_trained)

plt.plot(train_x)
#plt.plot(out_max)
plt.plot(train_y)
#plt.plot(out)
plt.plot(out_trained)
plt.show()

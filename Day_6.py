import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

print(glob.glob('DB/*.csv')) #DB폴더 안에 있는 csv확장자를 모두 찾아서 출력

train_x = []
train_y = []
PARA = ['UHOLEG1', 'UHOLEG2', 'UHOLEG3', 'ZINST58']
for one_file in glob.glob('DB/*.csv'):
    LOCA = pd.read_csv(one_file)
    if len(train_x) == 0:
        train_x = LOCA.loc[:, PARA].to_numpy()
        train_y = LOCA.loc[:, ['Normal_0']].to_numpy()
    else:
        get_x = LOCA.loc[:, PARA].to_numpy()
        get_y = LOCA.loc[:, ['Normal_0']].to_numpy()
        train_x = np.vstack((train_x, get_x))   #따로 있는 SHAPE을 하나로 합쳐줌 =(21,4)와 (21,4)가 있을때 하나로 합치면 (42,4)가 된다
        train_y = np.vstack((train_y, get_y))
    print(f'X_SHAPE : {np.shape(train_x)} ㅣ '
          f'Y_SHAPE : {np.shape(train_y)}')      #print안의 값이 바뀌는경우 f' : {}'
print('DONE')

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(np.shape(train_x)[1]),
    tf.keras.layers.Dense(200),
    tf.keras.layers.Dense(np.shape(train_y)[1]+1, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=150)

out_trained = model.predict(train_x[0:60])
plt.plot(train_y[0:60])
plt.plot(out_trained)
plt.show()
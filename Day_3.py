import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#데이터 받아오기.
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()

#데이터 모양 보기.
print(np.shape(mnist_data))
(x_train, y_train,), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
print(x_train[0])

#이미지 봐보기
plt.imshow(x_train[1])
plt.title(y_train[1])
plt.show()

#레이어 설계
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #Flatten은 2차원 함수를 1차원으로 바꿔줌
    tf.keras.layers.Dense(5, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='tanh')
])

out = model.predict([x_train[0:2]])
print(out)
print(np.shape(out))
#--------------------------------
print(y_train[0], type(y_train[0]))
print(x_train[0], type(x_train[0]))

#softmax는 확률적으로 가장 높은 값을 선택해줌
#make one hot
print(max(y_train))
temp_y = []
for one_y_val in y_train: #y_train(6만개 존재) 안에서 첫번째 꺼에 저장-> 두번째꺼에 저장 -> 세번째거에 저장...... 반복, one_y_val은 y_train에 입력되어있는 값들
    zero_array = np.zeros(10) #zero_array라는 공간을 만들고 10개의 0이 들어감 : zero_array = [0 0 0 0 0 0....]
    zero_array[one_y_val] = 1
    # one_y_val의 값이 해당하는 위치에 1이라는 설정 저장.
    # ex)첫번째 반복일때 : y_train 첫번째 값이 1 이면 zero_array=[0 1 0 0 0 0...], 두번째 반복일때 : y_train 두번째 값이 5이면 zero_array=[0 0 0 0 0 1 0 0 0 0]
    temp_y.append(zero_array) #save기능. 위에서 한 결과를 temp_y의 공간에 저장
temp_y = np. array(temp_y)
print(type(temp_y))
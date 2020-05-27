import pandas as pd

temp = pd.read_csv('Data_1.csv')
print(temp)

print(temp['Mon'])

temp['T'] = 0
print(temp['T'])

import matplotlib.pyplot as plt
plt.plot(temp['A'], dashes=[6, 2])
plt.xlim((0,5))
plt.ylim((0,10))
plt.show()
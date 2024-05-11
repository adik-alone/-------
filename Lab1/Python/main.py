import numpy as np
import matplotlib.pyplot as plt
k = 3

E = np.array([])
D = np.array([])
for i in range(10000):
    data = np.random.chisquare(k, 1000)
    # plt.hist(data, bins=50,  color='skyblue', edgecolor='black')
    # plt.show()
    e_loc = sum(data) / 1000
    E = np.append(E, e_loc)

plt.hist(E, bins=50,  color='skyblue', edgecolor='black')
plt.show()





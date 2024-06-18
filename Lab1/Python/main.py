import numpy as np
import matplotlib.pyplot as plt
k = 5

E = np.array([])
D = np.array([])
M = np.array([])

counting_samples = 1000
size_sample = 1000


for i in range(counting_samples):
    if i == 1:
        plt.hist(data, bins=50,  color='skyblue', edgecolor='black')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.title('Гистограмма выборки по распределению хи-квадрат')
        plt.savefig('hist-chi-square.png')
        # plt.show()
        plt.clf()
    data = np.random.chisquare(k, size_sample)
    data = np.sort(data)
    # e_loc = sum(data) / 1000
    e_loc = np.mean(data)
    d_loc = np.var(data, ddof=1)
    m_loc = np.median(data)
    M = np.append(M, m_loc)
    D = np.append(D, d_loc)
    E = np.append(E, e_loc)


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(E, bins=15,  color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Средние')
# plt.savefig('hist-mean.png')
# plt.show()
# plt.clf()

plt.subplot(1, 3, 2)
plt.hist(D, bins=15,  color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Дисперсии')
# plt.savefig('hist-disp.png')
# plt.show()
# plt.clf()

plt.subplot(1, 3, 3)
plt.hist(M, bins=15,  color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Квантели порядка 0.5')
# plt.savefig('hist-median.png')
plt.savefig('first-exp.png')
plt.show()
# plt.clf()





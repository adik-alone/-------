import numpy as np
import matplotlib.pyplot as plt
k = 3

E = np.array([])
D = np.array([])
M = np.array([])
for i in range(10000):
    if i == 1:
        plt.hist(data, bins=50,  color='skyblue', edgecolor='black')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.title('Гистограмма выборки по распределению хи-квадрат')
        plt.savefig('hist-chi-square.pdf')
        # plt.show()
        plt.clf()
    data = np.random.chisquare(k, 10000)
    data = np.sort(data)
    # e_loc = sum(data) / 1000
    e_loc = np.mean(data)
    d_loc = np.var(data, ddof=1)
    m_loc = np.median(data)
    M = np.append(M, m_loc)
    D = np.append(D, d_loc)
    E = np.append(E, e_loc)


plt.hist(E, bins=50,  color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Гистограмма срединх выборочных в 10000 выборках')
plt.savefig('hist-mean.pdf')
# plt.show()
plt.clf()

plt.hist(D, bins=50,  color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Гистограмма выборочных дисперсий в 10000 выборках')
plt.savefig('hist-disp.pdf')
# plt.show()
plt.clf()

plt.hist(M, bins=50,  color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Гистограмма выборочных квантелей порядка 0.5 в 10000 выборках')
plt.savefig('hist-median.pdf')
# plt.show()
plt.clf()


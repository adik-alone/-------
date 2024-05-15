import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, chi

data_gamma = np.random.gamma(2, 1, 10000)
plt.hist(data_gamma, bins=50,  color='skyblue', edgecolor='black')
# plt.savefig('gamma.png')
# plt.show()
plt.clf()

data_chi = np.random.chisquare(5, 10000)
plt.hist(data_chi, bins=50,  color='skyblue', edgecolor='black')
# plt.savefig('chi.png')
# plt.show()
plt.clf()

shape = 2
rate = 1
data_gamma = np.random.gamma(2, 1, 10000)

distrib_func_values = []
values = []
for i in range(1000):
    distribution = chi(4)
    samples = np.sort(distribution.rvs(1000))
    sec_value = samples[1]
    distrib_func_value = (distribution.cdf(sec_value) * 1000)
    values.append(distrib_func_value)
distrib_func_values.append(values)

hist, bins = np.histogram(distrib_func_values, bins=30, density=True)
plt.bar(bins[:-1], hist, width=np.diff(bins), color='skyblue', alpha=0.7, label='nF(X_2)')

x = np.linspace(0, 20, 100)
y = gamma.pdf(x, a=shape, scale=1 / rate)

plt.plot(x, y, color='red', linestyle='--', label='Г(2,1)')
plt.legend()
plt.tight_layout()
plt.savefig('second-two-one.png')
plt.show()




for i in range(1000):
    distribution = chi(4)
    samples = np.sort(distribution.rvs(1000))
    last_value = samples[-1]
    distrib_func_value = ((1 - distribution.cdf(last_value)) * 1000)
    values.append(distrib_func_value)
distrib_func_values.append(values)

hist, bins = np.histogram(distrib_func_values, bins=30, density=True)
plt.bar(bins[:-1], hist, width=np.diff(bins), color='skyblue', alpha=0.7, label='n(1 - F(X_n))')

shape = 1
rate = 1

x = np.linspace(0, 10, 50)
y = gamma.pdf(x, a=shape, scale=1 / rate)

plt.plot(x, y, color='red', linestyle='--', label='Г(1,1)')
plt.legend()
plt.tight_layout()
plt.savefig('second-one-one.png')
plt.show()





# plt.hist(data_gamma, bins=50,  color='skyblue', edgecolor='black')
# plt.savefig('gamma.png')
# plt.show()

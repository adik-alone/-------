import numpy as np
import matplotlib.pyplot as plt

data_gamma = np.random.gamma(2, 1, 10000)
plt.hist(data_gamma, bins=50,  color='skyblue', edgecolor='black')
plt.savefig('gamma.png')
plt.show()

data_chi = np.random.chisquare(5, 10000)
plt.hist(data_chi, bins=50,  color='skyblue', edgecolor='black')
plt.savefig('chi.png')
plt.show()



data_gamma = np.random.gamma(1, 1, 10000)
plt.hist(data_gamma, bins=50,  color='skyblue', edgecolor='black')
plt.savefig('gamma.png')
plt.show()

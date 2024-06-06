import numpy as np
import scipy.stats as stats

# Параметры задачи
n1, n2 = 10000, 10000
mu1, mu2 = 0, 0
sigma1, sigma2 = 2, 1
alpha = 0.05
total = 0
probability = 0


for i in range(1000):
    # генерация выборок
    X1 = np.random.normal(mu1, np.sqrt(sigma1), n1)
    X2 = np.random.normal(mu2, np.sqrt(sigma2), n2)

    # выборочные дисперсии
    S1 = np.var(X1, ddof=1)
    S2 = np.var(X2, ddof=1)

    tau = S1 / S2
    tau_real = sigma1 / sigma2

    # криты распределения фишера
    F_low = stats.f.ppf(alpha / 2, n1 - 1, n2 - 1)
    F_high = stats.f.ppf(1 - alpha / 2, n1 - 1, n2 - 1)

    # доверительный интервал
    CI_low = tau / F_high
    CI_high = tau / F_low

    if CI_low <= tau_real <= CI_high:
        total += 1

    probability = total / 1000


print(total, probability)

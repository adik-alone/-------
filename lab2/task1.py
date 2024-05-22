import numpy as np
import scipy.stats as stats

# Параметры задачи
n1, n2 = 25, 25
mu1, mu2 = 0, 0
sigma1_sq, sigma2_sq = 2, 1
alpha = 0.05

# Генерация выборок
X1 = np.random.normal(mu1, np.sqrt(sigma1_sq), n1)
X2 = np.random.normal(mu2, np.sqrt(sigma2_sq), n2)

# Выборочные дисперсии
S1_sq = np.var(X1, ddof=1)
S2_sq = np.var(X2, ddof=1)

# Оценка отношения дисперсий
tau_hat = (n2 * S1_sq) / (n1 * S2_sq)

# Критические значения распределения Фишера
F_low = stats.f.ppf(alpha / 2, dfn=n1-1, dfd=n2-1)
F_high = stats.f.ppf(1 - alpha / 2, dfn=n1-1, dfd=n2-1)

# Доверительный интервал
CI_low = tau_hat / F_high
CI_high = tau_hat / F_low

print(f"Доверительный интервал для отношения дисперсий: ({CI_low}, {CI_high})")

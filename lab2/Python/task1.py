import numpy as np
import scipy.stats as stats

# Параметры задачи
n1, n2 = 25, 25
mu1, mu2 = 0, 0
sigma1, sigma2 = 2, 1
alpha = 0.05

# Генерация выборок
# Генерирует массив чисел, распределенных по нормальному закону
X1 = np.random.normal(mu1, np.sqrt(sigma1), n1)
X2 = np.random.normal(mu2, np.sqrt(sigma2), n2)

# Выборочные дисперсии
# ddof=1, чтобы получить несмещённую оценку дисперсии (деление на n-1)
S1 = np.var(X1)
S2 = np.var(X2)

# Оценка отношения дисперсий
tau_hat = (n2 * S1) / (n1 * S2)
print(f"Оценка отношения дисперсий: {tau_hat}")

# Критические значения распределения Фишера
F_low = stats.f.ppf(alpha / 2, dfn=n1-1, dfd=n2-1)
F_high = stats.f.ppf(1 - alpha / 2, dfn=n1-1, dfd=n2-1)

# Доверительный интервал
CI_low = tau_hat / F_high
CI_high = tau_hat / F_low

print(f"Доверительный интервал для отношения дисперсий: ({CI_low}, {CI_high})")

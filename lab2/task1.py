import numpy as np
import scipy.stats as stats

# Параметры задачи
n1, n2 = 25, 25
mu1, mu2 = 0, 0
sigma1, sigma2 = 2, 1
alpha = 0.05
total = []

for i in range(1000):
    # Генерация выборок
    # Генерирует массив чисел, распределенных по нормальному закону
    X1 = np.random.normal(mu1, sigma1, n1)
    X2 = np.random.normal(mu2, sigma2, n2)

    # Выборочные дисперсии
    S1 = sum(X1 ** 2)
    S2 = sum(X2 ** 2)

    # Оценка отношения дисперсий
    tau = (n2 * S1) / (n1 * S2)
    print(f"Оценка отношения дисперсий: {tau}")

    # Критические значения распределения Фишера
    F_low = stats.f.ppf(alpha / 2, dfn=n1 - 1, dfd=n2 - 1)
    F_high = stats.f.ppf(1 - alpha / 2, dfn=n1 - 1, dfd=n2 - 1)
    print(f"F_low - {F_low}, F_high - {F_high}")

    # Доверительный интервал
    CI_low = tau / F_high
    CI_high = tau / F_low

    if CI_high >= tau >= CI_low:
        total.append(1)
    else:
        total.append(0)

    print(f"Доверительный интервал для отношения дисперсий: ({CI_low}, {CI_high})")

print(sum(total))

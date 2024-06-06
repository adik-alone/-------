import numpy as np
import scipy.stats as stats

p = 0.7
alpha = 0.05
n1 = 25
n2 = 10000
total1 = 0
total2 = 0
probability1 = 0
probability2 = 0

for i in range(1000):
    X1 = np.random.geometric(p, n1)
    X2 = np.random.geometric(p, n2)

    # оценка мат ожидания
    p_hat1 = 1 / np.mean(X1)
    p_hat2 = 1 / np.mean(X2)

    se1 = np.sqrt((1 - p_hat1) / (n1 * p_hat1 ** 2))
    se2 = np.sqrt((1 - p_hat2) / (n2 * p_hat2 ** 2))

    crit = stats.norm.ppf(1 - alpha / 2)

    ci_lower1 = p_hat1 - crit * se1
    ci_upper1 = p_hat1 + crit * se1

    ci_lower2 = p_hat2 - crit * se2
    ci_upper2 = p_hat2 + crit * se2

    if ci_lower1 <= p <= ci_upper1:
        total1 += 1

    if ci_lower2 <= p <= ci_upper2:
        total2 += 1

    probability1 = total1 / 1000
    probability2 = total2 / 1000


print(f"попавшие - {total1}, точность - {probability1}")
print(f"попавшие - {total2}, точность - {probability2}")

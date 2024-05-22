import numpy as np
from scipy.stats import norm

# Параметры
p_true = 0.7  # истинное значение параметра p
n = 100  # размер выборки
alpha = 0.05  # уровень значимости

# Генерация выборки из геометрического распределения
# np.random.geometric генерирует числа в диапазоне [1, ∞), поэтому нужно сдвинуть
sample = np.random.geometric(p_true, size=n)

# Вычисление оценочной вероятности успеха p
mean_X = np.mean(sample)
p_hat = 1 / mean_X

# Стандартная ошибка оценки
se_p_hat = np.sqrt(p_hat * (1 - p_hat) / (n * (1 - p_hat)**2))

# Квантиль стандартного нормального распределения для заданного уровня значимости
z = norm.ppf(1 - alpha / 2)

# Доверительный интервал
ci_lower = p_hat - z * se_p_hat
ci_upper = p_hat + z * se_p_hat

print(f'Оценка p: {p_hat}')
print(f'Стандартная ошибка: {se_p_hat}')
print(f'Доверительный интервал для p: ({ci_lower}, {ci_upper})')

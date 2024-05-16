import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла
df = pd.read_csv('mobile_phones.csv', delimiter=",", skiprows=1, names=['battery_power','blue','clock_speed','dual_sim',
                                                            'fc','four_g','int_memory','m_dep','mobile_wt','n_cores',
                                                            'pc','px_height','px_width','ram','sc_h','sc_w','talk_time',
                                                            'three_g','touch_screen','wifi','price_range'])

print(df[['dual_sim', 'n_cores', 'three_g']])

dual_sim_cnt = sum(df['dual_sim'])
three_g_cnt = sum(df['three_g'] == 1)
max_cores = max(df['n_cores'])

print("Количество моделей с двумя симкартами:", dual_sim_cnt)
print("Количество моделей поддерживающих 3g:", three_g_cnt)
print("Наибольшее число ядер у процессора:", max_cores)

srednee_battery_power = df['battery_power'].mean()
dispersia_battery_power = df['battery_power'].var()
mediana_battery_power = df['battery_power'].median()
kvantil_battery_power = df['battery_power'].quantile(2 / 5)

print("Выборочное среднее емкости аккумулятора:", srednee_battery_power)
print("Выборочная дисперсия емкости аккумулятора:", dispersia_battery_power)
print("Выборочная медиана емкости аккумулятора:", mediana_battery_power)
print("Выборочная квантиль порядка 2/5 емкости аккумулятора:", kvantil_battery_power)


with_wifi = df[df['wifi'] == 1]['battery_power']
without_wifi = df[df['wifi'] == 0]['battery_power']

# empericheskaya func
plt.figure(figsize=(10, 5))
plt.hist(df['battery_power'], bins=30, density=True, cumulative=True, histtype='step', label='Все телефоны', color='blue')
plt.xlabel('Емкость аккумулятора')
plt.ylabel('Эмпирическая функция распределения')
plt.title('График эмпирической функции распределения емкости аккумулятора')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(with_wifi, bins=30, density=True, cumulative=True, histtype='step', label='С Wi-Fi', color='green')
plt.xlabel('Емкость аккумулятора')
plt.ylabel('Эмпирическая функция распределения')
plt.title('График эмпирической функции распределения емкости аккумулятора')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(without_wifi, bins=30, density=True, cumulative=True, histtype='step', label='Без Wi-Fi', color='red')
plt.xlabel('Емкость аккумулятора')
plt.ylabel('Эмпирическая функция распределения')
plt.title('График эмпирической функции распределения емкости аккумулятора')
plt.legend()
plt.grid(True)
plt.show()

# gistogramma
plt.figure(figsize=(10, 5))
plt.hist(df['battery_power'], bins=30, alpha=0.7, label='Все телефоны', color='blue')
plt.xlabel('Емкость аккумулятора')
plt.ylabel('Частота')
plt.title('Гистограмма емкости аккумулятора')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(with_wifi, bins=30, alpha=0.7, label='С Wi-Fi', color='green')
plt.xlabel('Емкость аккумулятора')
plt.ylabel('Частота')
plt.title('Гистограмма емкости аккумулятора')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(without_wifi, bins=30, alpha=0.7, label='Без Wi-Fi', color='red')
plt.xlabel('Емкость аккумулятора')
plt.ylabel('Частота')
plt.title('Гистограмма емкости аккумулятора')
plt.legend()
plt.grid(True)
plt.show()

# box plot
plt.figure(figsize=(10, 5))
plt.boxplot([df['battery_power'], with_wifi, without_wifi], labels=['Все телефоны', 'С Wi-Fi', 'Без Wi-Fi'])
plt.ylabel('Емкость аккумулятора')
plt.title('Box-plot для емкости аккумулятора')
plt.grid(True)
plt.show()

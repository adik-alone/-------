import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

data = pd.read_csv('fifa_players_stats.csv', delimiter=",", skiprows=1, names=[
    'Known As', 'Full Name', 'Overall', 'Potential', 'Value(in Euro)', 'Positions Played',
    'Best Position', 'Nationality', 'Image Link', 'Age', 'Height(in cm)', 'Weight(in kg)', 'TotalStats',
    'BaseStats', 'Club Name', 'Wage(in Euro)', 'Release Clause', 'Club Position', 'Contract Until',
    'Club Jersey Number', 'Joined On', 'On Loan', 'Preferred Foot', 'Weak Foot Rating', 'Skill Moves',
    'International Reputation', 'National Team Name', 'National Team Image Link', 'National Team Position',
    'National Team Jersey Number', 'Attacking Work Rate', 'Defensive Work Rate', 'Pace Total', 'Shooting Total',
    'Passing Total', 'Dribbling Total', 'Defending Total', 'Physicality Total', 'Crossing', 'Finishing',
    'Heading Accuracy', 'Short Passing', 'Volleys', 'Dribbling', 'Curve', 'Freekick Accuracy', 'LongPassing',
    'BallControl', 'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping',
    'Stamina',
    'Strength', 'Long Shots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
    'Marking', 'Standing Tackle', 'Sliding Tackle', 'Goalkeeper Diving', 'Goalkeeper Handling', 'GoalkeeperKicking',
    'Goalkeeper Positioning', 'Goalkeeper Reflexes', 'ST Rating', 'LW Rating', 'LF Rating', 'CF Rating', 'RF Rating',
    'RW Rating', 'CAM Rating', 'LM Rating', 'CM Rating', 'RM Rating', 'LWB Rating', 'CDM Rating', 'RWB Rating',
    'LB Rating', 'CB Rating', 'RB Rating', 'GK Rating'])

data['total_rating'] = data['ST Rating'] + data['LW Rating'] + data['LF Rating'] + data['CF Rating'] + \
                       data['RF Rating'] + data['RW Rating'] + data['CAM Rating'] + data['LM Rating'] + data[
                           'CM Rating'] + data['RM Rating'] + \
                       data['LWB Rating'] + data['CDM Rating'] + data['RWB Rating'] + data['LB Rating'] + data[
                           'CB Rating'] + \
                       data['RB Rating'] + data['GK Rating']

columns_to_max = ['ST Rating', 'LW Rating', 'LF Rating', 'CF Rating', 'RF Rating', 'RW Rating', 'CAM Rating',
                  'LM Rating', 'CM Rating', 'RM Rating', 'LWB Rating', 'CDM Rating', 'RWB Rating', 'LB Rating',
                  'CB Rating', 'RB Rating', 'GK Rating']
data['max_rating'] = data[columns_to_max].max(axis=1)

# y = x beta + b
X = data[['total_rating', 'max_rating', 'Age']].values
y = data['Value(in Euro)'].values

n, k = X.shape

X = np.column_stack([X, np.ones(n)])

# коэффициенты
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# остаточная дисперсия
y_pred = X @ beta.T
N = n - k
dispersia = ((y - y_pred) ** 2).sum() / N
print("oстаточная дисперсия:", dispersia)


# коэффициент детерминации
y_srednee = y.mean()
total_sum = ((y - y_srednee) ** 2).sum()
total_sum_ostatki = ((y_pred - y_srednee) ** 2).sum()
r_square = 1 - (total_sum_ostatki / total_sum)
print("коэффициент детерминации (R^2):", r_square)


# доверительные интервалы
alpha = 0.05
t_value = stats.t.ppf(1 - alpha / 2, N)
st_er = np.sqrt(dispersia * np.diag(np.linalg.inv(X.T @ X)))
intervals = np.array([beta - t_value * st_er, beta + t_value * st_er]).T
print("доверительные интервалы:\n", intervals)


t_values = beta / st_er
p_values = stats.t.cdf(np.abs(t_values), N) * 2


# Гипотеза: Чем меньше возраст, тем больше цена
print(f"гипотеза 0: Чем меньше возраст, тем больше цена")
print(f"гипотеза 1: Чем меньше возраст, тем меньше цена")
print(f"t-статистика: {t_values[3]}, p-значение: {p_values[3]}")
if p_values[3] > alpha:
    print('0 гипотеза принята\n')
else:
    print('1 гипотеза принята\n')

# Гипотеза: Цена зависит от рейтинга
print(f"гипотеза 0: цена зависит от общего рейтинга")
print(f"гипотеза 1: цена не зависит от общего рейтинга")
print(f"t-статистика: {t_values[1]}, p-значение: {p_values[1]}")
if p_values[1] > alpha:
    print('0 гипотеза принята\n')
else:
    print('1 гипотеза принята\n')

# Гипотеза: Цена одновременно зависит от максимального рейтинга и возраста
print(f"гипотеза 0: цена одновременно зависит от максимального рейтинга и возраста")
print(f"гипотеза 1: цена одновременно не зависит от максимального рейтинга и возраста")
print(f"t-статистика возраста: {t_values[3]}, p-значение возраста: {p_values[3]}")
print(f"t-статистика максимального рейтинга: {t_values[2]}, p-значение максимального рейтинга: {p_values[2]}")
if p_values[3] > alpha and p_values[2] > alpha:
    print('0 гипотеза принята\n')
else:
    print('1 гипотеза принята\n')


plt.plot(data['total_rating'], data['Value(in Euro)'], 'o', color='red')
plt.xlabel('Общий рейтинг')
plt.ylabel('Цена')
plt.title('Цена от общего рейтинга')
# plt.savefig('total_rating.png')
plt.show()

plt.plot(data['max_rating'], data['Value(in Euro)'], 'o', color='green')
plt.xlabel('Максимальный рейтинг')
plt.ylabel('Цена')
plt.title('Цена от макс рейтинга')
# plt.savefig('max_rating.png')
plt.show()

plt.plot(data['Age'], data['Value(in Euro)'], 'o', color='orange')
plt.xlabel('Возраст')
plt.ylabel('Цена')
plt.title('Цена от возраста')
# plt.savefig('age.png')
plt.show()

import math
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


# С помощью критерия согласия Пирсона хи-квадрат проверить согласованность
# рейтинг футболиста с нормальным законом (формализовать основные и альтернативные ги-
# потезы, реализовать самостоятельно).

# гипотезы:
# 0 гипотеза - нормальное распределение
# 1 гипотеза - не нормальное распределение
def ratings(data):
    # Ожидаемые значения
    srednee = data['total_rating'].mean()
    otklonenie = data['total_rating'].std()

    observed, bins = np.histogram(data['total_rating'], bins='auto')
    expected = np.array([stats.norm.cdf(bins[i + 1], srednee, otklonenie) -
                         stats.norm.cdf(bins[i], srednee, otklonenie) for i in
                         range(len(bins) - 1)]) * len(data['total_rating'])
    expected = expected * (observed.sum() / expected.sum())

    xi_square_nabl = ((observed - expected) ** 2 / expected).sum()
    print(xi_square_nabl)
    xi_square_crit = stats.chi2.ppf(0.95, math.ceil(math.log2(data.shape[0])+1)-3)
    print(xi_square_crit)

    plt.figure(figsize=(10, 6))
    plt.hist(data['total_rating'], bins=bins, label='Observed', color='blue')
    plt.plot((bins[1:] + bins[:-1]) / 2, expected, 'o-', label='Expected', color='red')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Chi-Squared Test for Normality')
    plt.legend()
    plt.show()

    if xi_square_crit > xi_square_nabl:
        print(f"0 гипотеза (нормальное распределение)")
    else:
        print(f"1 гипотеза (ненормальное распределение)")

    return xi_square_nabl

# Ту же самую задачу решить с помощью другого
# критерия (тоже формализовать гипотезы, но здесь можно воспользоваться готовой реализацией)


# Задание 2: Проверка однородности и построение графика
def xi_square_ages(data, age):
    young = data[data['Age'] < age]['total_rating']
    old = data[data['Age'] >= age]['total_rating']

    observed_young, bins = np.histogram(young, bins='auto')
    observed_old, old_bins = np.histogram(old, bins=bins)
    observed_young = observed_young * (observed_old.sum() / observed_young.sum())

    xi_square_nabl = stats.chi2_contingency([observed_young, observed_old])

    plt.figure(figsize=(10, 6))
    plt.hist(young, bins=bins, label='Young', color='blue')
    plt.hist(old, bins=bins, label='Old', color='green')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Chi-Squared Test for Homogeneity')
    plt.legend()
    plt.show()

    return xi_square_nabl


# Задание 3: Проверка независимости и построение графика
# def chi_squared_test_independence(data):
#     contingency_table = pd.crosstab(data['total_rating'], data['Nationality'])
#     chi_stat = stats.chi2_contingency(contingency_table)
#
#     plt.figure(figsize=(10, 6))
#     contingency_table.plot(kind='bar', stacked=True)
#     plt.xlabel('total_rating')
#     plt.ylabel('Count')
#     plt.title('Chi-Squared Test for Independence')
#     plt.show()
#
#     return chi_stat


ratings(data)

age = 30
xi_square = xi_square_ages(data, age)


# chi_stat = chi_squared_test_independence(data)
# print(f'Статистика хи-квадрат: {chi_stat}')

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
    print(f'наблюдаемое значение хи-квадрат критерия: {xi_square_nabl}')
    xi_square_crit = stats.chi2.ppf(0.95, math.ceil(math.log2(data.shape[0])+1)-3)
    print(f'критическое значение хи-квадрат критерия: {xi_square_crit}')




    plt.figure(figsize=(10, 6))
    plt.hist(data['total_rating'], bins=bins, label='Observed', color='blue')
    plt.plot((bins[1:] + bins[:-1]) / 2, expected, 'o-', label='Expected', color='red')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Chi-Squared Test for Normality')
    plt.legend()
    # plt.savefig('task1.png')
    plt.show()

    if xi_square_crit > xi_square_nabl:
        print(f"0 гипотеза (нормальное распределение)")
    else:
        print(f"1 гипотеза (ненормальное распределение)")

    return xi_square_nabl

# Ту же самую задачу решить с помощью другого
# критерия (тоже формализовать гипотезы, но здесь можно воспользоваться готовой реализацией)
def auto_checker_rating(data):
    # Проверяем выборку на нормальность c помощью критерия Колмогорова-Смирнова
    statistic, pvalue = stats.kstest(data["total_rating"], 'norm')

    # Уровень значимости
    alpha = 0.05
    if pvalue > alpha:
        print("Гипотеза о нормальности распределения не отвергается.")
    else:
        print("Гипотеза о нормальности распределения отвергается.")


# Задание 2: Проверка однородности и построение графика
def xi_square_ages(data, age):
    print("\n")
    print("Проверка гипотезы о том, что выборки молодых и возрастных футболистов однородны:")
    # Гипотеза 0 - выборки молодых и возрастных футболистов однородны
    # Гипотеза 1 - выборки молодых и возрастных футболистов не однородны

    young = data[data['Age'] < age]['total_rating']
    old = data[data['Age'] >= age]['total_rating']

    observed_young, bins = np.histogram(young, bins='auto')
    observed_old, old_bins = np.histogram(old, bins=bins)
    observed_young = observed_young * (observed_old.sum() / observed_young.sum())
    count_str_1 = observed_young.shape[0]
    count_str_2 = observed_old.shape[0]

    xi_square_nabl = stats.chi2_contingency([observed_young, observed_old])[0]
    # xi_square_nabl = stats.chi2_contingency([young, old])[0]
    print(f'наблюдаемое значение критерия: {xi_square_nabl}')

    # подсчёт количества степеней свободы
    df = (count_str_1 - 1) * (count_str_2 - 1)
    xi_square_crit = stats.chi2.ppf(0.95, df)
    print(f'критическое значение критерия: {xi_square_crit}')

    plt.figure(figsize=(10, 6))
    plt.hist(young, bins=bins, label='Young', color='blue')
    plt.hist(old, bins=bins, label='Old', color='green')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Chi-Squared Test for Homogeneity')
    plt.legend()
    plt.show()

    if xi_square_crit > xi_square_nabl:
        print("нет оснований отвергать нулевую гипотезу -> выборки однородны")
    else:
        print("есть основания отвергать нулевую гипотезу -> выборки не однородны")

    auto_checker_age(young, old)

    return xi_square_nabl

def auto_checker_age(sample1, sample2):
    # Критерий Колмогорова-Смирнова
    statistic, pvalue = stats.ks_2samp(sample1, sample2)


    print("\n")
    print("Проверка гипотезы о том, что выборки молодых и возрастных футболистов однородны с помощью критерия Колмогорова-Смирнова:")
    print(f"Статистика Колмогорова-Смирнова: {statistic}")
    print(f"p-значение: {pvalue}")

    alpha = 0.05
    if pvalue < alpha:
        print("Отвергаем нулевую гипотезу: выборки неоднородны.")
    else:
        print("Нет оснований отвергать нулевую гипотезу: выборки однородны.")

# Задание 3: Проверка независимости и построение графика
# def chi_squared_test_independence(data):
#     ratings = data['total_rating']
#     nations = data['Nationality']
#
#     # observed_nation, nation_bins = np.histogram(nations, bins='auto')
#     data_for = {
#         'rating': ratings,
#         'nation': nations
#     }
#     df = pd.DataFrame(data_for)
#     contingency_table = pd.crosstab(df['rating'], df['nation'])
#
#     chi2, p_value, dof, expected = stats.chi2_contingency(ratings, nations)
#
#     # Выводим результаты теста
#     print("Наблюдаемое значение критерия хи-квадрат:", chi2)
#     print("p-значение:", p_value)
#     print("Степени свободы:", dof)
#     print("Ожидаемые частоты:", expected)
#
#
#     #С помощью критерия Фишера
#     odds_ratio, p_val = stats.fisher_exact(contingency_table)
#
#     # Вывод результатов
#     print("Отношение шансов (odds ratio):", odds_ratio)
#     print("p-значение:", p_val)
#
#
#     # observed_rating, bins = np.histogram(ratings, bins='auto')
#     # observed_nation, nation_bins = np.histogram(nation, bins=bins)
#     # observed_rating = observed_rating * (observed_nation.sum() / observed_rating.sum())
#     #
#     # # contingency_table = pd.crosstab(data['total_rating'], data['Nationality'])
#     # chi_stat = stats.chi2_contingency(contingency_table)
#     # # chi_stat = stats.chi2_contingency([data['total_rating'], data['Nationality']])[0]
#     #
#     #
#     # plt.figure(figsize=(10, 6))
#     # contingency_table.plot(kind='bar', stacked=True)
#     # plt.xlabel('total_rating')
#     # plt.ylabel('Count')
#     # plt.title('Chi-Squared Test for Independence')
#     # plt.show()
#
#     return chi_stat

def task3(data):

    k = math.ceil(math.log2(data.shape[0]) + 1)

    contingency_table = pd.crosstab(data['Nationality'], pd.cut(data['total_rating'], k))

    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)
    total_sum = contingency_table.sum().sum()
    expected = np.outer(row_sums, col_sums)

    chi_square = total_sum * sum([((contingency_table.iloc[i, j] - expected[i, j] / total_sum) ** 2) / expected[i, j] for i in range(contingency_table.shape[0]) for j in range(contingency_table.shape[1])])

    alpha = 0.05
    df = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)
    chi_square_crit = stats.chi2.ppf(1 - alpha, df)
    print('xi^2:', chi_square)
    print('xi_crit^2:', chi_square_crit)

    if chi_square <= chi_square_crit:
        print("Гипотеза H0 принимается")
    else:
        print("Гипотеза H0 отвергается")


    # # Критерий фишера
    # data['rating_group'] = pd.cut(data['total_rating'], 2, labels=['Small', 'Much'])
    #
    # # Таблица сопряженности
    # contingency_table = pd.crosstab(data['rating_group'], data['Nationality'])
    #
    # # Критерий Фишера
    # odds_ratio, pvalue = stats.fisher_exact(contingency_table)
    # print(f"p-значение: {pvalue}")
    #
    # # Интерпретация:
    # if pvalue < 0.05:
    #     print("Гипотеза о независимости отвергается. Существует зависимость между возрастом и городом.")
    # else:
    #     print("Гипотеза о независимости не отвергается. Недостаточно доказательств для утверждения о зависимости.")


    # print("Отношение шансов (odds ratio):", odds_ratio)
    # print("p-значение:", pvalue)


    # plt.plot([i.mid for i in
    #           contingency_table.columns], contingency_table.iloc[0, :])
    # plt.plot([i.mid for i in
    #           contingency_table.columns], contingency_table.iloc[1, :])
    # plt.legend()
    # plt.show()

    # ratings = data['total_rating']
    # nations = data['Nationality']
    #
    # #категоризация выборки:
    #
    # data['rating_groupe'] = pd.qcut(data['total_rating'], q = 100)
    #
    #
    # #нахождение наблюдаемого значения
    #
    # #
    #
    #
    #
    #
    # # Пример данных: числовой параметр и категориальный параметр
    #
    # # Создание таблицы сопряженности
    # observed_data = pd.crosstab(ratings, nations)
    #
    # # Применение критерия Хи-квадрат
    # chi2, p, _, _ = stats.chi2_contingency(observed_data)
    #
    # # Вывод результатов
    # print("Хи-квадрат статистика:", chi2)
    # print("p-значение:", p)
    #

ratings(data)
print("\n")
print("Критерий Колмогорова-Смирнова")
auto_checker_rating(data)

age = 30
xi_square = xi_square_ages(data, age)

print("\n")
task3(data)
# chi_stat = chi_squared_test_independence(data)
# print(f'Статистика хи-квадрат: {chi_stat}')

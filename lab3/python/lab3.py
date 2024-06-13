import numpy as np
import pandas as pd
import scipy.stats as stats

df = pd.read_csv('fifa_players_stats.csv', delimiter=",", skiprows=1, names=[
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


# Критерий хи-квадрат Пирсона
def chi_square_test(data):
    # Ожидаемые значения
    expected_freq, bins = np.histogram(data, bins='auto', density=True)
    expected_freq *= len(data) * np.diff(bins)

    # Наблюдаемые значения
    observed_freq, _ = np.histogram(data, bins=bins)

    # Статистика хи-квадрат
    chi_square_stat = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()

    # Степени свободы
    dof = len(observed_freq) - 1

    # p-value
    p_value = 1 - stats.chi2.cdf(chi_square_stat, df=dof)

    return chi_square_stat, p_value


chi_square_stat, p_value = chi_square_test(df)
print(f'Chi-square Statistic: {chi_square_stat}, p-value: {p_value}')

# Критерий Колмогорова-Смирнова
ks_stat, ks_p_value = stats.kstest(df, 'norm', args=(df.mean(), df.std()))
print(f'KS Statistic: {ks_stat}, p-value: {ks_p_value}')

# Генерация данных для возрастов и рейтингов
ages = np.random.randint(18, 40, size=100)
young_ratings = df[ages < 30]
old_ratings = df[ages >= 30]

# Формализация гипотез
# H0: Рейтинги молодых и возрастных футболистов однородны
# H1: Рейтинги молодых и возрастных футболистов не однородны

# Объединение данных
observed_freq = np.array([
    np.histogram(young_ratings, bins='auto')[0],
    np.histogram(old_ratings, bins='auto')[0]
])

# Ожидаемые значения
expected_freq = observed_freq.sum(axis=0) * observed_freq.sum(axis=1)[:, None] / observed_freq.sum()

# Статистика хи-квадрат
chi_square_stat = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()
dof = (observed_freq.shape[0] - 1) * (observed_freq.shape[1] - 1)
p_value = 1 - stats.chi2.cdf(chi_square_stat, df=dof)

print(f'Chi-square Statistic: {chi_square_stat}, p-value: {p_value}')

# Генерация данных для национальностей и рейтингов
nationalities = np.random.choice(['A', 'B', 'C'], size=100)
ratings_by_nationality = pd.crosstab(nationalities, df)

# Формализация гипотез
# H0: Рейтинг и национальность независимы
# H1: Рейтинг и национальность не независимы

# Критерий хи-квадрат
chi_square_stat, p_value, _, _ = stats.chi2_contingency(ratings_by_nationality)

print(f'Chi-square Statistic: {chi_square_stat}, p-value: {p_value}')

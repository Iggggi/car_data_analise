import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# https://www.kaggle.com/austinreese/craigslist-carstrucks-data
data_frame = pd.read_csv("./vehicles.csv")

# вывод столбцов и их типов
column_types=data_frame.dtypes
print(column_types)

# output
# id                int64
# url              object
# region           object
# region_url       object
# price             int64
# year            float64
# manufacturer     object
# model            object
# condition        object
# cylinders        object
# fuel             object
# odometer        float64
# title_status     object
# transmission     object
# VIN              object
# drive            object
# size             object
# type             object
# paint_color      object
# image_url        object
# description      object
# county          float64
# state            object
# lat             float64
# long            float64
# posting_date     object
# dtype: object

# чистка информацийй удаление строк с пропущенной датой и ценой
data_frame.drop(data_frame[data_frame['price'] == 0].index, inplace=True)
data_frame.drop(data_frame[data_frame['year'].isna()].index, inplace=True)
data_frame.drop(data_frame[data_frame['year'] == 2021].index, inplace=True)
data_frame.drop([496], inplace=True)
data_frame.drop(data_frame[data_frame.odometer > 500000].index, inplace=True)

# приведение типа данных year к int
data_frame['year'] = data_frame['year'].astype(int)
data_frame['year'].dtype

# Imputing missing values in odometer with median odometer of each year cars
# data_frame['odometer'] = data_frame['odometer'].fillna(data_frame['year'].apply(lambda x: year_med.get(x)))

# Удаление оставшихся пропущенных значений в одометре, поскольку информация за эти годы недоступна
data_frame.drop(data_frame[data_frame['odometer'].isna()].index, inplace=True)

# удаление значений с ценой менее 1500 пробегом менее 50,000 и годом выпуска старше 2010
data_frame.drop(
    data_frame[(data_frame['price'] < 1500) & (data_frame['odometer'] < 50000) & (data_frame['year'] > 2010)].index,
    inplace=True)
data_frame.drop(data_frame[(data_frame['price'] < 200)].index, inplace=True)
data_frame.drop(data_frame[(data_frame['price'] > 50000)].index, inplace=True)

# Установка количества цилиндров для производителя tesla cars равным 0
data_frame.loc[data_frame.manufacturer == 'tesla', 'cylinders'] = 0
data_frame.cylinders.fillna(0, inplace=True)
data_frame.cylinders = data_frame.cylinders.replace('other', 0)

# количество машин по состоянию good/excelent/new/like new/fair/salvage
data_frame.condition.value_counts()

# количество машин по типу топлива
data_frame.fuel.value_counts()

# построение матрицы корреляций: цена имеет положительную корреляцию с годом выпуска и отрицательную с пробегом
plt.figure(figsize=(8, 6))
spearman = data_frame.corr(method = 'spearman')
sns.heatmap(spearman, annot = True)

# анализ распределений
sns.pairplot(data_frame)
plt.show()

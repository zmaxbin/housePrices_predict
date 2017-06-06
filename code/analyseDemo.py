# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("D:/Kaggle/House_prices/data/train.csv")
df_test = pd.read_csv('D:/Kaggle/House_prices/data/test.csv')

# 查看数据属性
# print df_train.columns

#descriptive statistics summary
# df_train['SalePrice'].describe()

# 画直方图
# sns.distplot(df_train['SalePrice'],kde=False)
# sns.plt.show()

# 绘制峰度和偏度
# print("Skewness: %f" % df_train['SalePrice'].skew())
# print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.show()
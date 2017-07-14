# -*- coding:utf-8 -*-
# 2017/2/13 21:18

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv('../data/test.csv',index_col = 0)


train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('Un')
size_mapping = {
    'Ex': 6,
    'Gd': 5,
    'TA': 4,
    'Fa':3,
    'Po':2,
    'Un':1}
train_df['FireplaceQu'] = train_df['FireplaceQu'].map(size_mapping)
# 箱型图
sns.boxplot(y='SalePrice',x='FireplaceQu',data=train_df)
plt.show()

total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(40)
# PoolQC         1453  0.995205
# MiscFeature    1406  0.963014
# Alley          1369  0.937671
# Fence          1179  0.807534

# missing data
# total = test_df.isnull().sum().sort_values(ascending=False)
# percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(40)
#               Total   Percent
# PoolQC         1456  0.997944
# MiscFeature    1408  0.965045
# Alley          1352  0.926662
# Fence          1169  0.801234
# FireplaceQu     730  0.500343
# LotFrontage     227  0.155586

# var = 'LotFrontage'
# data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()

train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('Un')
size_mapping = {
    'Ex': 6,
    'Gd': 5,
    'TA': 4,
    'Fa':3,
    'Po':2,
    'Un':1}
train_df['FireplaceQu'] = train_df['FireplaceQu'].map(size_mapping)
# 散点图
# var = 'FireplaceQu'
# data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()

# 箱型图
# sns.boxplot(y='SalePrice',x='FireplaceQu',data=train_df)
# plt.show()

# from sklearn.linear_model import RANSACRegressor,LinearRegression
# ransac = RANSACRegressor(LinearRegression(),
#                          max_trials=100,
#                          min_samples=50,
#                          residual_metric=lambda x:np.sum(np.abs(x),axis=1),
#                          residual_threshold=300000,
#                          random_state=0)
# train_df = train_df.fillna(train_df.mean())
# X = train_df[['LotFrontage']].values
# y = train_df['SalePrice'].values
# ransac.fit(X,y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
# line_X = np.arange(3,300,1)
# line_y_ransac = ransac.predict(line_X[:,np.newaxis])
# plt.scatter(X[inlier_mask],y[inlier_mask],
#             c='blue',marker='o',label='Inliers')
# plt.scatter(X[outlier_mask],y[outlier_mask],
#             c='lightgreen',marker='s',label='Outliers')
# plt.plot(line_X,line_y_ransac,color='red')
# plt.xlabel('FireplaceQu')
# plt.ylabel('SalePrice')
# plt.legend(loc='upper letf')
# plt.show()

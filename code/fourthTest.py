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
# 缺失值分析
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('Un')
size_mapping1 = {
    'Ex': 6,
    'Gd': 5,
    'TA': 4,
    'Fa':3,
    'Po':2,
    'Un':1}
train_df['FireplaceQu'] = train_df['FireplaceQu'].map(size_mapping1)
# 将异常值用均数替代
train_df['LotFrontage'].replace(to_replace=313,value=train_df['LotFrontage'].mean(),inplace=True)
train_df['LotFrontage'].replace(to_replace=313,value=train_df['LotFrontage'].mean(),inplace=True)
# 去除缺失值太大的项
train_df = train_df.drop((missing_data[missing_data['Total'] > 1178]).index,1)
test_df = test_df.drop((missing_data[missing_data['Total'] > 1168]).index,1)
# 将异常值替换为均值
train_df['BsmtFinSF1'].replace(to_replace=5644,value=train_df['BsmtFinSF1'].mean(),inplace=True)
train_df['TotalBsmtSF'].replace(to_replace=6110,value=train_df['TotalBsmtSF'].mean(),inplace=True)
train_df['1stFlrSF'].replace(to_replace=4692,value=train_df['1stFlrSF'].mean(),inplace=True)
train_df['OpenPorchSF'].replace(to_replace=523,value=train_df['OpenPorchSF'].mean(),inplace=True)
train_df['EnclosedPorch'].replace(to_replace=552,value=train_df['EnclosedPorch'].mean(),inplace=True)

train_df['LotArea'].replace(to_replace=215245,value=train_df['LotArea'].mean(),inplace=True)
train_df['LotArea'].replace(to_replace=164660,value=train_df['LotArea'].mean(),inplace=True)
train_df['LotArea'].replace(to_replace=159000,value=train_df['LotArea'].mean(),inplace=True)
train_df['LotArea'].replace(to_replace=115149,value=train_df['LotArea'].mean(),inplace=True)


train_df['GrLivArea'].replace(to_replace=5642,value=train_df['GrLivArea'].mean(),inplace=True)
train_df['GrLivArea'].replace(to_replace=4676,value=train_df['GrLivArea'].mean(),inplace=True)
# train_df = train_df.drop(train_df[train_df['Id'] == 524].index)
train_df = train_df.drop(train_df[train_df['Id'] == 309].index)
train_df = train_df.drop(train_df[train_df['Id'] == 633].index)
train_df = train_df.drop(train_df[train_df['Id'] == 1324].index)

all_data = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))

# 填充缺失值与数字化
size_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa':2,
    'Po':1}

size_mapping2 = {
    'Ex': 6,
    'Gd': 5,
    'TA': 4,
    'Fa':3,
    'Po':2,
    'NA':1}

train_df['ExterQual'] = train_df['ExterQual'].map(size_mapping)
train_df['ExterCond'] = train_df['ExterCond'].map(size_mapping)
train_df['BsmtQual'] = train_df['BsmtQual'].map(size_mapping2)
train_df['BsmtCond'] = train_df['BsmtCond'].map(size_mapping2)
train_df['HeatingQC'] = train_df['HeatingQC'].map(size_mapping)
train_df['KitchenQual'] = train_df['KitchenQual'].map(size_mapping)
train_df['GarageQual'] = train_df['GarageQual'].map(size_mapping2)
train_df['GarageCond'] = train_df['GarageCond'].map(size_mapping2)

# 相关系数矩阵
train_df = pd.get_dummies(train_df)

#filling NA's with the mean of the column:
train_df = train_df.fillna(train_df.mean())

#log transform the target:
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# 残差图
from sklearn.model_selection import train_test_split
X = train_df.iloc[:,:-1].values
y = train_df['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
slr = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='red',marker='s',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=10,xmax=20,lw=1,color='black')
plt.xlim([10,20])
plt.show()

a = (y_train_pred-y_train)
b = (max(y_train_pred-y_train))
c = (list(y_train_pred-y_train).index(max(list(y_train_pred-y_train))))
print(y_train_pred[658]-y_train[658])
print(y_train_pred[658])
print y_train[658]
print(np.exp(y_train[658]))
print np.log1p(np.exp(y_train[658]))
print ""


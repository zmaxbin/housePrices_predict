# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

train_df = pd.read_csv("D:/Kaggle/housePrices_predict/data/train.csv")
test_df = pd.read_csv('D:/Kaggle/housePrices_predict/data/test.csv',index_col = 0)


# DecisioinTree
# def lin_regplot(X, y, model):
#     plt.scatter(X, y, c='lightblue')
#     plt.plot(X, model.predict(X), color='red', linewidth=2)
#     return
#
# X = train_df[['OverallQual']].values
# y = train_df['SalePrice'].values
# tree = DecisionTreeRegressor(max_depth=3)
# tree.fit(X,y)
# sort_idx = X.flatten().argsort()
# lin_regplot(X[sort_idx],y[sort_idx],tree)
# plt.xlabel('% Overall material and finish quality')
# plt.ylabel('SalePrice')
# plt.show()

# RFs
#数据预处理
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(20)

df_train = train_df.drop((missing_data[missing_data['Total'] > 250]).index,1)

total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(20)

test_df = test_df.drop((missing_data[missing_data['Total'] > 200]).index,1)

# df_train['BsmtFinSF1'].replace(to_replace=5644,value=df_train['BsmtFinSF1'].mean(),inplace=True)
# df_train['TotalBsmtSF'].replace(to_replace=6110,value=df_train['TotalBsmtSF'].mean(),inplace=True)
# df_train['1stFlrSF'].replace(to_replace=4692,value=df_train['1stFlrSF'].mean(),inplace=True)
#
# df_train['LotArea'].replace(to_replace=215245,value=df_train['LotArea'].mean(),inplace=True)
# df_train['LotArea'].replace(to_replace=164660,value=df_train['LotArea'].mean(),inplace=True)
# df_train['LotArea'].replace(to_replace=159000,value=df_train['LotArea'].mean(),inplace=True)
# df_train['LotArea'].replace(to_replace=115149,value=df_train['LotArea'].mean(),inplace=True)
#
#
# df_train['GrLivArea'].replace(to_replace=5642,value=df_train['GrLivArea'].mean(),inplace=True)
# df_train['GrLivArea'].replace(to_replace=4676,value=df_train['GrLivArea'].mean(),inplace=True)
#
# df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

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

all_data['ExterQual'] = all_data['ExterQual'].map(size_mapping)
all_data['ExterCond'] = all_data['ExterCond'].map(size_mapping)
all_data['BsmtQual'] = all_data['BsmtQual'].map(size_mapping2)
all_data['BsmtCond'] = all_data['BsmtCond'].map(size_mapping2)
all_data['HeatingQC'] = all_data['HeatingQC'].map(size_mapping)
all_data['KitchenQual'] = all_data['KitchenQual'].map(size_mapping)
all_data['GarageQual'] = all_data['GarageQual'].map(size_mapping2)
all_data['GarageCond'] = all_data['GarageCond'].map(size_mapping2)

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

# 算法部分
# X = all_data.iloc[:, :-1].values
# y = all_data['SalePrice'].values

X_train = all_data[:df_train.shape[0]]
X_test = all_data[df_train.shape[0]:]
y_train = df_train.SalePrice

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_final = np.exp(forest.predict(X_test))
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('D:/Kaggle/housePrices_predict/data/submission7.csv',columns = ['Id','SalePrice'],index = False)


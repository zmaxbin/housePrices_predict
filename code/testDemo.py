# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv("D:/Kaggle/housePrices_predict/data/train.csv")
test_df = pd.read_csv('D:/Kaggle/housePrices_predict/data/test.csv',index_col = 0)

# missing data
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(20)

test_df = test_df.drop((missing_data[missing_data['Total'] > 200]).index,1)

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(20)

df_train = train_df.drop((missing_data[missing_data['Total'] > 250]).index,1)
# print df_train.isnull().sum().max()

#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# concat函数相当于拼接，拼接方式是增加行数，不增加列数
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
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

#creating matrices for sklearn:
X_train = all_data[:df_train.shape[0]]
X_test = all_data[df_train.shape[0]:]
y = df_train.SalePrice

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")



model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
y_final = np.exp(model_lasso.predict(X_test))
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('E:/Kaggle/housePrices_predict/data/submission3.csv',columns = ['Id','SalePrice'],index = False)
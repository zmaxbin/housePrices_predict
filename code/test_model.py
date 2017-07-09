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

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv('../data/test.csv',index_col = 0)

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

# corrmat = train_df.corr()
# print corrmat.nlargest(37,'SalePrice')['SalePrice']


#scatter plot grlivarea/saleprice
# var = 'EnclosedPorch'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()

#deleting points
# my = df_train.sort_values(by = 'TotalBsmtSF', ascending = False)[:1]
df_train['BsmtFinSF1'].replace(to_replace=5644,value=df_train['BsmtFinSF1'].mean(),inplace=True)
df_train['TotalBsmtSF'].replace(to_replace=6110,value=df_train['TotalBsmtSF'].mean(),inplace=True)
df_train['1stFlrSF'].replace(to_replace=4692,value=df_train['1stFlrSF'].mean(),inplace=True)

df_train['LotArea'].replace(to_replace=215245,value=df_train['LotArea'].mean(),inplace=True)
df_train['LotArea'].replace(to_replace=164660,value=df_train['LotArea'].mean(),inplace=True)
df_train['LotArea'].replace(to_replace=159000,value=df_train['LotArea'].mean(),inplace=True)
df_train['LotArea'].replace(to_replace=115149,value=df_train['LotArea'].mean(),inplace=True)


df_train['GrLivArea'].replace(to_replace=5642,value=df_train['GrLivArea'].mean(),inplace=True)
df_train['GrLivArea'].replace(to_replace=4676,value=df_train['GrLivArea'].mean(),inplace=True)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# df_train = df_train.drop(df_train[df_train['Id'] == 333].index)

# print df_train.describe().T

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

model_lassoCV = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
y_pred_lassoCV = model_lassoCV.predict(X_test)

from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.001,random_state=0).fit(X_train,y)
y_pred_lasso = model_lasso.predict(X_test)


from sklearn.kernel_ridge import KernelRidge
model_ridge = KernelRidge(alpha=10).fit(X_train,y)
y_pred_ridge = model_ridge.predict(X_test)



# from sklearn.model_selection import validation_curve
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Lasso
#
# pipe_lr = Pipeline([
#                     ('clf', KernelRidge())])
#
# param_range = [0.01,0.1,1,10,100,1000]
# train_scores, test_scores = validation_curve(
#                 estimator=pipe_lr,
#                 X=X_train,
#                 y=y,
#                 param_name='clf__alpha',
#                 param_range=param_range,
#                 cv=10)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.plot(param_range, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='training accuracy')
#
# plt.fill_between(param_range, train_mean + train_std,
#                  train_mean - train_std, alpha=0.15,
#                  color='blue')
#
# plt.plot(param_range, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='validation accuracy')
#
# plt.fill_between(param_range,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter alpha')
# plt.ylabel('Accuracy')
# plt.ylim([0.8, 1.0])
# plt.tight_layout()
# # plt.savefig('./figures/validation_curve.png', dpi=300)
# plt.show()



import xgboost as xgb

regr = xgb.XGBRegressor(
    colsample_bytree=0.2,
    gamma=0.0,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=1.5,
    n_estimators=7200,
    reg_alpha=0.9,
    reg_lambda=0.6,
    subsample=0.2,
    seed=42,
    silent=1)

regr.fit(X_train, y)


# Run prediction on the Kaggle test set.
y_pred_xgb = regr.predict(X_test)

y_final = np.exp((y_pred_xgb+y_pred_lassoCV+y_pred_lasso+y_pred_ridge)/4)

submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('../data/submission8.csv',columns = ['Id','SalePrice'],index = False)


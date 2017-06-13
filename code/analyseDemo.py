# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
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

df_train['ExterQual'] = df_train['ExterQual'].map(size_mapping)
df_train['ExterCond'] = df_train['ExterCond'].map(size_mapping)
df_train['BsmtQual'] = df_train['BsmtQual'].map(size_mapping2)
df_train['BsmtCond'] = df_train['BsmtCond'].map(size_mapping2)
df_train['HeatingQC'] = df_train['HeatingQC'].map(size_mapping)
df_train['KitchenQual'] = df_train['KitchenQual'].map(size_mapping)
df_train['GarageQual'] = df_train['GarageQual'].map(size_mapping2)
df_train['GarageCond'] = df_train['GarageCond'].map(size_mapping2)

df_train = pd.get_dummies(df_train)

#filling NA's with the mean of the column:
df_train = df_train.fillna(df_train.mean())

#creating matrices for sklearn:
# X_train = all_data[:df_train.shape[0]]
# X_test = all_data[df_train.shape[0]:]
# y = df_train.SalePrice
# 残差图
X = df_train.iloc[:,:-1].values
y = df_train['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
slr = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
# from sklearn.model_selection import cross_val_score
#
# def rmse_cv(model):
#     rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
#     return(rmse)
#
# alphas = [1, 3,5,10,30,50,70,100]
# cv_Lasso = [rmse_cv(Lasso(alpha = alpha)).mean()
#             for alpha in alphas]
#
# cv_Lasso = pd.Series(cv_Lasso, index = alphas)
# cv_Lasso.plot(title = "Validation - Just Do It")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()
# print cv_Lasso
# print cv_Lasso.min()


#
# print y_train[391]
# print (y_train_pred-y_train)
# print min(y_train_pred-y_train)
# print list(y_train_pred-y_train).index(min(list(y_train_pred-y_train)))
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='red',marker='s',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=10,xmax=20,lw=1,color='black')
plt.xlim([10,20])
plt.show()

# 均方误差
print 'SalePrice train:%.3f,test:%.3f' %(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred))


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
y_final = np.exp(model_lasso.predict(X_test))
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('D:/Kaggle/housePrices_predict/data/submission2.csv',columns = ['Id','SalePrice'],index = False)


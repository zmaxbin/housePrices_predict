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
test_df = test_df.drop(test_df.loc[test_df['Electrical'].isnull()].index)

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print missing_data.head(20)

df_train = train_df.drop((missing_data[missing_data['Total'] > 250]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# print df_train.isnull().sum().max()


# concat函数相当于拼接，拼接方式是增加行数，不增加列数
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

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
print test_df.index
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('D:/Kaggle/House_prices/data/submission2.csv',columns = ['Id','SalePrice'],index = False)





# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# 文件的组织形式是house price文件夹下面放house_price.py和input文件夹
# input文件夹下面放的是从https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data下载的train.csv  test.csv  sample_submission.csv 和 data_description.txt 四个文件

# step1 检查源数据集，读入数据，将csv数据转换为DataFrame数据
train_df = pd.read_csv("D:/Kaggle/House_prices/data/train.csv",index_col = 0)
test_df = pd.read_csv('D:/Kaggle/House_prices/data/test.csv',index_col = 0)

# step2 合并数据，进行数据预处理
prices = pd.DataFrame({'price':train_df['SalePrice'],'log(price+1)':np.log1p(train_df['SalePrice'])})

y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df,test_df),axis = 0)


# step3 变量转化
print all_df['MSSubClass'].dtypes
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
print all_df['MSSubClass'].dtypes
print all_df['MSSubClass'].value_counts()
# 把category的变量转变成numerical表达形式
# get_dummies方法可以帮你一键one-hot
print pd.get_dummies(all_df['MSSubClass'],prefix = 'MSSubClass').head()
all_dummy_df = pd.get_dummies(all_df)
print all_dummy_df.head()

# 处理好numerical变量
print all_dummy_df.isnull().sum().sort_values(ascending = False).head(11)
# 我们这里用mean填充
mean_cols = all_dummy_df.mean()
print mean_cols.head(10)
all_dummy_df = all_dummy_df.fillna(mean_cols)
print all_dummy_df.isnull().sum().sum()

# 标准化numerical数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print numeric_cols
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std

# step4 建立模型
# 把数据处理之后，送回训练集和测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
print dummy_train_df.shape,dummy_test_df.shape

# 将DF数据转换成Numpy Array的形式，更好地配合sklearn

X_train = dummy_train_df.values
X_test = dummy_test_df.values

ridge = Ridge(alpha = 15)
rf = RandomForestRegressor(n_estimators = 500,max_features = .3)
ridge.fit(X_train,y_train)
rf.fit(X_train,y_train)

y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

y_final = (y_ridge + y_rf) / 2

# Step 6: 提交结果
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
print submission_df.head(10)
submission_df.to_csv('D:/Kaggle/House_prices/data/submission.csv',columns = ['Id','SalePrice'],index = False)
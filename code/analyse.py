# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv("D:/Kaggle/housePrices_predict/data/train.csv")
test_df = pd.read_csv('D:/Kaggle/housePrices_predict/data/test.csv',index_col = 0)

# print train_df.dtypes[train_df.dtypes != "object"].index
# sns.set(style='whitegrid',context='notebook')
# cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea']
# sns.pairplot(train_df[cols],size=2.5)
# plt.show()

# corrmat = train_df.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.xticks(rotation=90)
# plt.show()

# 散点图
#scatter plot grlivarea/saleprice
# var = 'LotFrontage'
# data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()

#saleprice correlation matrix
k = 15 #number of variables for heatmap
corrmat = train_df.corr()
# index1 = list(((corrmat.nlargest(37,'SalePrice')['SalePrice'].values)<0.111447)).count(True)
# index2 = list(((corrmat.nlargest(37,'SalePrice')['SalePrice'].values)>-0.077856)).count(True)
print corrmat.nlargest(37,'SalePrice')['SalePrice'].index[25:33]

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea','TotalBsmtSF']
sns.pairplot(train_df[cols], size = 2.5)
plt.show()


print corrmat.nlargest(37,'SalePrice')['SalePrice'].index[0:24]
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.xticks(rotation=90)
plt.show()
# -*- coding:utf-8 -*-

from numpy import array
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

train_df = pd.read_csv("D:/Kaggle/House_prices/data/train.csv",index_col = 0)
test_df = pd.read_csv('D:/Kaggle/House_prices/data/test.csv',index_col = 0)
#
# total= train_df.isnull().sum().sort_values(ascending=False)
# percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# print missing_data.head(20)


imr = Imputer(missing_values='NaN',strategy='median',axis=0)
imr = imr.fit(array(train_df.values[:,2]).reshape(1460,1))
imputed_data = imr.transform(array(train_df.values[:,2]).reshape(1460,1))
# print imputed_data

lenths = array(imputed_data)
max_x = max(lenths)
min_x = min(lenths)

pyplot.hist(lenths,200)

pyplot.xlabel('LotFrontage')
pyplot.xlim(0,320)
pyplot.ylabel('Frequency')
pyplot.title('Histogram Of LotFrontage')
pyplot.grid(True)
pyplot.show()

# draw_hist(lenths)
#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

train = pd.read_csv('data/sales_train.csv')

train_dat = train.drop('date', axis=1)
train_dat = train.groupby(['date_block_num','shop_id','item_id'],).agg({'item_price':'mean','item_cnt_day':'sum'})
train_dat.reset_index(inplace=True)
print(train_dat['date_block_num'])
Xy_train = train_dat.loc[train_dat['date_block_num'] < 33, :]
Xy_test = train_dat.loc[train_dat['date_block_num'] == 34, :]

print(Xy_train)
print(Xy_test)



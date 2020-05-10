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
Xy_test = train_dat.loc[train_dat['date_block_num'] == 33, :]

print(Xy_train)
print(Xy_test)

y_test_pred = []
n = 0

print(Xy_test.shape)
for i, row in Xy_test.iterrows():
    n += 1
    #print(n)
    if(n > 10): break
    ShopItemSet = Xy_train.loc[(Xy_train['shop_id'] == row['shop_id']) & (Xy_train['item_id'] == row['item_id']),:].copy()

    if(ShopItemSet.empty):
        y_test_pred.append(0.0)
    else:
        model = LinearRegression()
        model.fit(ShopItemSet[['date_block_num']], ShopItemSet['item_cnt_day'])
        x_test = pd.DataFrame(
                {'date_block_num':[row['date_block_num']]})
        y_preds = model.predict(x_test)
        y_test_pred.append(y_preds[0])


print(y_test_pred)
print(Xy_test['item_cnt_day'][0:10])



pred = pd.DataFrame({'real':Xy_test['item_cnt_day'],'pred':y_test_pred})


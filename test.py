#!/usr/bin/python3

import pandas as pd
import numpy as np

training_file = "data/sales_train.csv"
train = pd.read_csv(training_file)

train_dat = train.groupby(['date_block_num','shop_id'],).agg({'item_price':'mean','item_cnt_day':'sum'})


print(train_dat.values)

train_dat.to_csv('test.csv')

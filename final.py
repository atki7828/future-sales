#!/usr/bin/python3

import numpy as np
import pandas as pd

training_file = "data/sales_train.csv"
shops_file = "data/shops.csv"
items_file = "data/items.csv"
itemcat_file = "data/item_categories.csv"

train_data = pd.read_csv(training_file)

shops_id = pd.read_csv(shops_file).values[:,1]
items_id = pd.read_csv(items_file).values[:,1:2]
cat_id = pd.read_csv(itemcat_file).values[:,1]

# column 1 is the month number, from 0 to 33.
print(train_data['date_block_num'])
months = train_data['date_block_num'].unique()
print(train_data.values)
print('months',months)

print('shops',shops_id)
print('items',items_id)
print('categories',cat_id)


'''
	training_file contains the number of every item sold by every shop on certain dates.
	I think we need to find the number of items sold per month, so we'll have to add
	up the number for individual shops over every month.
	should store in 4d array:  [month_no,shop_id,item_no,items_sold]
'''

items_per_shop = []

print(train_data['date_block_num'])

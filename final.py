#!/usr/bin/python3

import os

if(not os.path.exists('./data/sales_train.csv')):
    import zipfile
    with zipfile.ZipFile('competitive-data-science-predict-future-sales.zip','r') as zp:
        zp.extractall('./data')
    print('files extracted')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_file = "data/sales_train.csv"
shops_file = "data/shops.csv"
items_file = "data/items.csv"
itemcat_file = "data/item_categories.csv"
items_per_shop_file = "data/items_per_shop.csv"

train_data = pd.read_csv(training_file)

shop_ids = pd.read_csv(shops_file).values[:,1]
item_ids = pd.read_csv(items_file).values[:,1:2]
cat_id = pd.read_csv(itemcat_file).values[:,1]

# column 1 is the month number, from 0 to 33.
months = train_data['date_block_num'].unique()

'''
	training_file contains the number of every item sold by every shop on certain dates.
        this function aggregates the dates and groups them by month: 
        month_num, shop_id, item_id, item_count
        for number of items shops sold per month.
        storing in new file: items_per_shop.csv.
'''

items_per_shop = []

def GetTotalItemCount(month,shop_id):
    return sum(train_data[(train_data['date_block_num']==month) & (train_data['shop_id'] == shop_id)]['item_cnt_day'])

def PlotShop(shop_id):
    shop_data = items_per_shop_df[items_per_shop_df['shop_id']==shop_id]
    month = shop_data.values[:,0]
    x = [i for i,_ in enumerate(month)]
    count = shop_data.values[:,2]
    plt.plot(x,count,label=str(shop_id))
    plt.xticks(x)
    plt.xlabel('month')
    plt.ylabel('items sold')
    plt.title('total items per month in shop ' + str(shop_id))
    plt.legend()
    plt.show()

def PlotAllShops():
    for shop_id in shop_ids:
        shop_data = items_per_shop_df[items_per_shop_df['shop_id'] == shop_id]
        month = shop_data.values[:,0]
        x = [i for i,_ in enumerate(month)]
        count = shop_data.values[:,2]
        plt.plot(x,count,label=str(shop_id))
        plt.xticks(x)
        plt.xlabel('month')
        plt.ylabel('items sold')
        plt.title('total items per month per shop')
        plt.legend()
    plt.show()

# getting the items per shop per month data.
print('aggregating shop data')
for month in months[0:2]:
    print('month {}/{}'.format(month,len(months)))
    for shop in shop_ids:
        row = [month,shop,GetTotalItemCount(month,shop)]
        items_per_shop.append(row)

items_per_shop_df = pd.DataFrame(items_per_shop,columns=['month_num','shop_id','item_count'])

items_per_shop_df.to_csv(items_per_shop_file,index=False)
'''
items_per_shop_df = pd.read_csv(items_per_shop_file)
'''
PlotAllShops()
for shop in shop_ids:
    PlotShop(shop)

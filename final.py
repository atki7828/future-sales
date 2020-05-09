#!/usr/bin/python3

import os

'''
https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview/description

description:
        We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.
'''

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
items_per_month_file = "data/items_per_month.csv"

train_data = pd.read_csv(training_file)

shop_ids = pd.read_csv(shops_file).values[:,1]
item_ids = pd.read_csv(items_file).values[:,1]
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
items_per_month = []

# returns number of items sold at shop_id in month
def GetItemsPerMonthCount(month,shop_id):
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
    if(not os.path.exists('./graphs/')):
        os.makedirs("./graphs/")
    plt.savefig('graphs/shop'+str(shop_id)+'.png')
    plt.close()

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
    plt.savefig('shops.png')
    plt.close()

# this function doesn't work yet.
def PlotItem(item_id):
    item_data = items_per_month_df[items_per_month_df['item_id']==item_id]
    month = item_data.values[:,0]
    x = [i for i,_ in enumerate(month)]
    count = item_data.values[:2]
    plt.plot(x,count,label=str(item_id))
    plt.xticks(x)
    plt.xlabel('month')
    plt.ylabel('number sold')
    plt.title('total number of item {} sold per month'.format(item_id))
    plt.legend()
    plt.show()
    if(not os.path.exists('./graphs/')):
        os.makedirs("./graphs/")
    plt.savefig('graphs/shop'+str(item_id)+'.png')
    plt.close()

# returns total number of item_id item sold in month
def GetTotalItemCount(month,item_id):
    return sum(train_data[(train_data['date_block_num']==month) & (train_data['item_id']==item_id)]['item_cnt_day'])

# getting the total items per shop per month data:
# month_num,shop_id,item_count
def ItemsPerShopPerMonth():
    for month in months:
        print('month {}/{}'.format(month,len(months)))
        for shop in shop_ids:
            row = [month,shop,GetItemsPerMonthCount(month,shop)]
            items_per_shop.append(row)

    items_per_shop_df = pd.DataFrame(items_per_shop,columns=['month_num','shop_id','item_count'])

    items_per_shop_df.to_csv(items_per_shop_file,index=False)

if not os.path.exists(items_per_shop_file):
    ItemsPerShopPerMonth()

# getting total number of each item sold per month.
# month_num,item_id,item_count
def TotalItemsPerMonth():
    for month in months:
        print('month {}/{}'.format(month,len(months)));
        month_data = train_data[train_data['date_block_num']==month]
        i = 0
        for item in month_data['item_id'].unique():
            #print('\titem {}/{}'.format(i,len(month_data['item_id'].unique())))
            i+=1
            row = [month,item,GetTotalItemCount(month,item)]
            items_per_month.append(row)
    items_per_month_df = pd.DataFrame(items_per_month,columns=['month_num','item_id','item_count'])
    items_per_month_df.to_csv(items_per_month_file,index=False)

'''
if not os.path.exists(items_per_shop_file):
    ItemsPerShopPerMonth()
else:
    items_per_shop_df = pd.read_csv(items_per_shop_file)
'''

if not os.path.exists(items_per_month_file):
    TotalItemsPerMonth()
else:
    items_per_month_df = pd.read_csv(items_per_month_file)
print(items_per_month_df)

#PlotItem(22154)

PlotAllShops()

#for shop in shop_ids:
#    PlotShop(shop)

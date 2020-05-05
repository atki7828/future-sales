#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd

items_per_shop_file = "data/items_per_shop.csv"

items_per_shop = pd.read_csv(items_per_shop_file);

#print(items_per_shop['shop_id'].unique())
#print(len(items_per_shop['shop_id'].unique()))

for shop_id in items_per_shop['shop_id'].unique():
    shop_data = items_per_shop[items_per_shop['shop_id']==shop_id]
    print(shop_data.values)
    month = shop_data.values[:,0]
    x = [i for i,_ in enumerate(month)]
    print(x)
    count = shop_data.values[:,2]
    plt.plot(x,count,label=str(shop_id))
    plt.xticks(x)
    plt.xlabel('month')
    plt.ylabel('items sold')
    plt.title('total items per month in shop ' + str(shop_id))
    plt.legend()
    plt.show()


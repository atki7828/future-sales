#!/usr/bin/python3

from keras.models import Sequential
from keras.optimizers import SGD 
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def color():
    return (
            random.uniform(0,0.8),
            random.uniform(0,0.8),
            random.uniform(0,0.8)
            )

train_file = "./data/sales_train.csv"
shops_file = "./data/shops.csv"
items_file = "./data/items.csv"

train = pd.read_csv('data/sales_train.csv')
shop_ids = pd.read_csv(shops_file).values[:,1]

train_data = train.groupby(['date_block_num','shop_id'],).agg({'item_cnt_day':'sum'})
#train_data = train.groupby(['date_block_num','item_id'],).agg({'item_cnt_day':'sum'})
train_data = train_data.reset_index()

x = train_data['shop_id']
y = train_data['item_cnt_day']
Xy_train = train_data[train_data['date_block_num'] < 33]
Xy_test = train_data[train_data['date_block_num'] == 33]
x_train = Xy_train[['shop_id','date_block_num']]
y_train = Xy_train[['item_cnt_day','date_block_num']]
x_test = Xy_test[['shop_id','date_block_num']]
y_test = Xy_test[['item_cnt_day','date_block_num']]
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
y_test_pred = []
n = 0
print(x_train.shape,y_train.shape)
x_train = np.transpose(x_train).values
y_train = np.transpose(y_train).values
print('xtrain',x_train)
print(x_test.shape)
x_test = np.transpose(x_test).values
print("xtest",x_test)
y_test = np.transpose(y_test).values

model = Sequential()
model.compile(optimizer='rmsprop',loss='mean_squared_error')
model.fit(x_train,y_train)

pred = model.predict(x_test)

#for shop_id in shop_ids:
for index,row in Xy_test.iterrows():
    shop_id = row['shop_id']
    n += 1
    print(n)
    #if(n > 100): break
    #ShopItemSet = Xy_train.loc[(Xy_train['shop_id'] == row['shop_id']) & (Xy_train['item_id'] == row['item_id']),:].copy()
    #ShopSet = Xy_train[Xy_train['shop_id'] == row['shop_id']].copy()
    ShopSet = Xy_train[Xy_train['shop_id'] == shop_id].copy()
    #ItemSet = Xy_train.loc[Xy_train['item_id'] == row['item_id'],:].copy()

    #if(ItemSet.empty):
    if(ShopSet.empty):
        y_test_pred.append(0.0)
    else:
        model.compile(optimizer= 'rmsprop',loss='mean_squared_error')
        print(ShopSet['date_block_num'].values)
        model.fit(ShopSet['date_block_num'].values.reshape(-1,1), ShopSet['item_cnt_day'].values)
        #model.fit(ItemSet[['date_block_num']], ItemSet['item_cnt_day'])
        x_test = pd.DataFrame(
                {'date_block_num':[row['date_block_num']]})
        print(x_test)
        y_preds = model.predict(x_test)
        y_pred = y_preds[0]
        #y_pred = 0 if y_pred < 0 else y_pred
        #y_pred = 20 if y_pred > 20 else y_pred
        y_test_pred.append(y_pred)


#for i in range(len(y_test_pred)):
#    print(str(y_test_pred[i])+'\t'+str(Xy_test['item_cnt_day'].values[i]))

#plt.plot(Xy_test['item_cnt_day'][0:len(y_test_pred)],y_test_pred,'o')
#plt.xlim([min(y_test_pred),max(y_test_pred)])
col = []
for i in range(len(y_test_pred)):
    col.append(color())

plt.figure(figsize=(20,10))
plt.scatter(Xy_test['shop_id'],y=y_test_pred,label='pred',c=col,marker='x')
plt.scatter(Xy_test['shop_id'],y=Xy_test['item_cnt_day'][0:len(y_test_pred)],label='real',c=col,marker='.')
plt.xticks(Xy_test['shop_id'])
plt.legend()
plt.savefig('prediction-scatter.png')
plt.close()
plt.figure(figsize=(20,10))
plt.bar(Xy_test['shop_id'],y_test_pred,label='pred',width=0.35,align='edge')
plt.bar(Xy_test['shop_id'],Xy_test['item_cnt_day'],label='real',width=-0.35,align='edge')
plt.xticks(Xy_test['shop_id'])
plt.legend()
plt.savefig('prediction-bar.png')


from sklearn.metrics import mean_squared_error

print('RMSE valid : %.3f' % \
        (np.sqrt(mean_squared_error(Xy_test['item_cnt_day'][0:len(y_test_pred)], y_test_pred))) )
for i in Xy_test['item_cnt_day']:
    if(i > 100):
        print(i)

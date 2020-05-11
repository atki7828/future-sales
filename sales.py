#!/usr/bin/python3

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import random

def color():
    return (
            random.uniform(0,0.8),
            random.uniform(0,0.8),
            random.uniform(0,0.8)
            )

def my_evaluation(y, y_pred):
    mse = sklm.mean_squared_error(y, y_pred)
    mse_log = sklm.mean_squared_log_error(y, y_pred)
    r2 = sklm.r2_score(y, y_pred)
    return mse, mse_log, r2

train_file = "./data/sales_train.csv"
shops_file = "./data/shops.csv"
items_file = "./data/items.csv"
test_file = './data/test.csv'

train = pd.read_csv('data/sales_train.csv')
shop_ids = pd.read_csv(shops_file)

train_data = train.groupby(['date_block_num','shop_id'],).agg({'item_cnt_day':'sum'})
#train_data = train.groupby(['date_block_num','item_id'],).agg({'item_cnt_day':'sum'})
train_data = train_data.reset_index()

x = train_data['shop_id']
y = train_data['item_cnt_day']
Xy_train = train_data[train_data['date_block_num'] < 33]
Xy_test = train_data[train_data['date_block_num'] == 33]

y_test_val = []
y_pred = []
n = 0

model = LinearRegression()

test = pd.read_csv(test_file)
test_dat = test.copy()
test_dat['date_block_num'] = 34
print(test_dat)
shop_test = shop_ids.copy()
shop_test['date_block_num'] = 34
shop_test = shop_test[['shop_id','date_block_num']]
print(shop_test)


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
        y_test_val.append(0.0)
    else:
        model.fit(ShopSet[['date_block_num']], ShopSet['item_cnt_day'])
        #model.fit(ItemSet[['date_block_num']], ItemSet['item_cnt_day'])
        x_test = pd.DataFrame(
                {'date_block_num':[row['date_block_num']]})
        print('xtest:',x_test['date_block_num'])
        y_vals = model.predict(x_test)
        y_val = y_vals[0]
        y_val = 0 if y_val < 0 else y_val
        #y_val = 20 if y_val > 20 else y_val
        y_test_val.append(y_val)

Xy_train = train_data.copy()

for index,row in shop_test.iterrows():
    shop_id = row['shop_id']
    shop_data = Xy_train[Xy_train['shop_id'] == shop_id].copy()
    m = shop_data.iloc[-1]['date_block_num']
    for i in range(int(m)+1,34):
        r = pd.DataFrame({'date_block_num':i,'shop_id':shop_id,'item_cnt_day':0},index=[i])
        shop_data = shop_data.append(r,ignore_index=True)

    model = LinearRegression()
    model.fit(shop_data[['date_block_num']],shop_data['item_cnt_day'])
    x_test = pd.DataFrame(
            {'date_block_num':[row['date_block_num']]})
    y_preds = model.predict(x_test)
    #print(shop_data)
    y = y_preds[0]
    y = 0 if y < 0 else y
    y_pred.append(y)
    s = Xy_train[Xy_train['shop_id']==shop_id].copy()
    x = range(34)
    plt.bar(s['date_block_num'],s['item_cnt_day'])
    plt.xticks(x)
    plt.bar(34,y)
    plt.title('month 34 prediction for shop {}'.format(shop_id))
    plt.savefig('./graphs/shop'+str(shop_id)+'pred.png')
    plt.close()


#for i in range(len(y_pred)):
#    print(i,y_pred[i])

#for i in range(len(y_test_val)):
#    print(str(y_test_val[i])+'\t'+str(Xy_test['item_cnt_day'].values[i]))

#plt.plot(Xy_test['item_cnt_day'][0:len(y_test_val)],y_test_val,'o')
#plt.xlim([min(y_test_val),max(y_test_val)])
col = []
for i in range(len(y_test_val)):
    col.append(color())

plt.figure(figsize=(20,10))
plt.scatter(Xy_test['shop_id'],y=y_test_val,label='pred',c=col,marker='x')
plt.scatter(Xy_test['shop_id'],y=Xy_test['item_cnt_day'][0:len(y_test_val)],label='real',c=col,marker='.')
plt.xticks(Xy_test['shop_id'])
plt.legend()
plt.savefig('prediction-scatter.png')
plt.close()
plt.figure(figsize=(20,10))
plt.bar(Xy_test['shop_id'],y_test_val,label='pred',width=0.35,align='edge')
plt.bar(Xy_test['shop_id'],Xy_test['item_cnt_day'],label='real',width=-0.35,align='edge')
plt.xticks(Xy_test['shop_id'])
plt.legend()
plt.title('prediction vs actual shop sales')
plt.savefig('prediction-bar.png')


plt.figure(figsize=(20,10))
plt.bar(range(len(y_pred)),y_pred)
plt.xticks(range(len(y_pred)))
plt.legend()
plt.savefig('prediction2-scatter.png')
plt.close()
from sklearn.metrics import mean_squared_error


mse,mse_log,r2 = my_evaluation(y_test_val,Xy_test['item_cnt_day'])
print('mse:',mse)
print('mse log',mse_log)
print('r2:',r2)
print('RMSE valid : %.3f' % \
        (np.sqrt(mean_squared_error(Xy_test['item_cnt_day'][0:len(y_test_val)], y_test_val))) )

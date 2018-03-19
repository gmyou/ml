import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb

path = './input/'

start_time = time.time()

train = pd.read_csv(path+"ct-sample_train_10000.csv")
train.columns = ['timestamp', 'sdk', 'platform', 'pub_idx', 'pub_app', 'pub_tagid', 'country', 'shop_idx', 'campaign_group_idx', 'campaign_idx', 'ads_idx', 'format_idx', 'ads_width', 'ads_height', 'adsize', 'bid', 'win', 'click', 'ctr']
test = pd.read_csv(path+"ct-sample_test.csv")

print('[{}] Finished to load data'.format(time.time() - start_time))

del train['sdk']
del train['platform']
del train['pub_app']
del train['country']
del train['adsize']
del train['pub_tagid']

del test['sdk']
del test['platform']
del test['pub_app']
del test['country']
del test['adsize']
del test['pub_tagid']

sub = pd.DataFrame()

sub['ads_idx'] = test['ads_idx']
sub['ctr'] = test['ctr']
# test.drop('ctr', axis=1, inplace=True)

print('[{}] Start XGBoost Training'.format(time.time() - start_time))


y = train['ctr']


x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)

print('[{}] Start XGBoost Training'.format(time.time() - start_time))


watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

print watchlist

params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'reg:linear', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True}

model = xgb.train(params, xgb.DMatrix(x1, y1), 260, watchlist, maximize=True, verbose_eval=10)

print('[{}] Start XGBoost Training'.format(time.time() - start_time))

print test

sub['ctr'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('./output/ct-sample_result.csv',index=False)

print sub
import pandas as pd
import tushare  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def rmse(y_true, y_pred):
    return np.sqrt(sum((np.asarray(y_pred) - np.asarray(y_true)) ** 2) / len(y_true))
    
    
def add_lag(data_now, window):
    lag_mean = data.rolling(window=window, closed='both').mean()
    lag_std = data.rolling(window=window, closed='both').std()
    lag_median = data.rolling(window=window, closed='both').median()
    ret = pd.merge(
        data_now, 
        pd.merge(
            lag_mean, 
            pd.merge(lag_std, lag_median, how='left', on='time'),
            how='left', on='time'
        ),
        how='left', on='time'
    )
    return ret


topic = pd.read_csv('./data/topic.csv', parse_dates=[0])
senti = pd.read_csv('./data/comments_senti_stats.csv', parse_dates=[0])
feature = pd.merge(topic, senti, how='inner', on='time')
feature = feature.sort_values(by='time')

stock = tushare.get_hist_data('sh', start='2018-05-18', end='2019-05-12')[['close', 'p_change']].reset_index()
stock['date'] = pd.to_datetime(stock['date'], format ='%Y-%m-%d')

data = pd.merge(feature, stock, how='left', left_on='time', right_on='date').dropna().\
    sort_values(by='time')
del data['date']
data = data.set_index('time')


combine = add_lag(add_lag(add_lag(data, 7), 14), 30).sort_values(by='time').shift(periods=2).dropna()
combine = pd.merge(combine, data[['close']], how='left', on='time')

clf = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01, max_depth=3,
    min_samples_leaf=9,
    criterion='mse'
).fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(rmse(y_train, y_pred_train), rmse(y_test, y_pred_test))

sep = int(combine.shape[0]*0.8)
predictions = []
for i in range(sep, combine.shape[0]):
    X_train = combine.iloc[:i, :]
    y_train = X_train.pop('close')
    X_test = combine.iloc[i:, :]
    y_test = X_test.pop('close')
    
    clf = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.01, max_depth=3,
        min_samples_leaf=9
    ).fit(X_train, y_train)
    predictions.append(clf.predict(X_test)[0])
    

plt.figure(figsize=(8, 6))
plt.plot(combine.close, 'r')
plt.plot(y_pred_train_s, 'g')
plt.plot(predictions, 'y')
plt.xlabel('time')
plt.ylabel('close')
plt.legend(['True', 'Train', 'Test'])

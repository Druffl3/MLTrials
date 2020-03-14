import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('dark_background')

quandl.ApiConfig.api_key = 'kV_u52hHJ67-nqsY4opz'  #for registered users
#df = quandl.get_table('WIKI/PRICES', date='2016-7-18') #getting data frame from quandl
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High', 'Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100 #high-low percentage change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100 #daily percentage change

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]  #Our Features!

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True) #fill NAN(not available values) (pandas)

forecast_out = int(math.ceil(0.009*len(df)))  #len(df) returns close to 3282, this line returns the number of days to 
                                             #push the column up, here 30 days.

df['label'] = df[forecast_col].shift(-forecast_out)

#Generally X=features, y=label

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)  #Scaling must be done with all the values together
X_lately = X[-forecast_out:] #The X value for which we will be testing to find y
X = X[:-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
#y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print (forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel = ('Date')
plt.ylabel = ('Price')
plt.show()
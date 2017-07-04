import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import cross_val_score

from pandas.tools.plotting import lag_plot,autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

import warnings
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import preprocessing




def create_stats(old_df,window):
	df = old_df.copy()
	for each_col_name in df.columns:
		df[each_col_name+"RollingMean"]=df[each_col_name].rolling(window=window).mean()
		#df[each_col_name+"DifferencedRollingMean"]=df[each_col_name+"RollingMean"]-df[each_col_name+"RollingMean"].shift()
		df[each_col_name+"RollingStd"]=df[each_col_name].rolling(window=window).std()
		df[each_col_name+"RelAvg"]=(df[each_col_name]/df[each_col_name+"RollingMean"])-1
		df[each_col_name+"RelAvgDifferenced"]=df[each_col_name+"RelAvg"]-df[each_col_name+"RelAvg"].shift()
		
		#if you pick this, you gotta drop twice the lenght of windows. Here we've got an average on an average
		df[each_col_name+"RelAvgRollingMean"]=df[each_col_name+"RelAvg"].rolling(window=window).mean()
		
		#df[each_col_name+"RelAvgRollingMeanDifferenced"]=df[each_col_name+"RelAvgRollingMean"]-df[each_col_name+"RelAvgRollingMean"].shift()
		df[each_col_name+"RollingStandardization"]=(df[each_col_name]-df[each_col_name+"RollingMean"])/df[each_col_name+"RollingStd"]
		#df[each_col_name+"RollingStandardizationDifferenced"]=df[each_col_name+"RollingStandardization"]-df[each_col_name+"RollingStandardization"].shift()
		df[each_col_name+"RollingNormalization"]=(df[each_col_name]-df[each_col_name+"RollingMean"])/(df[each_col_name].rolling(window=window).max()-df[each_col_name].rolling(window=window).min())
		#df[each_col_name+"RollingNormalizationDifferenced"]=df[each_col_name+"RollingNormalization"]-df[each_col_name+"RollingNormalization"].shift()
		
		df=df.drop(each_col_name,axis=1)
	
	#df=df.ix[(window)-1:];
	df=df.ix[(2*window)-1:]; # if RelAvgRollingMean is on, choose this one
	#df.dropna(inplace=True)
	df.fillna(0,inplace=True);
	return df

def chart_stats(df):
	nber_of_graphs=df.columns.shape[0]
	graphs_per_row=3
	nber_of_rows= int(nber_of_graphs/graphs_per_row)

	nber_of_rows_number=nber_of_rows*100
	graphs_per_row_number=graphs_per_row*10
	i=1
	base_number=nber_of_rows_number+graphs_per_row_number
	plt.rcParams['figure.figsize'] = (15, 3*nber_of_rows)
	for each_column in df.columns:    

		plt.subplot(base_number+i) 
		plt.plot(df[each_column])
		plt.title(each_column)
		i+=1
		plt.tight_layout()

	
	
	
from statsmodels.tsa.stattools import adfuller
	
def test_stationarity(timeseries):
	
	#Determing rolling statistics
	rolmean =timeseries.rolling(window=12,center=False).mean()
	rolstd = timeseries.rolling(window=12,center=False).std()
	#rolmean = pd.rolling_mean(timeseries, window=12)
	#rolstd = pd.rolling_std(timeseries, window=12)

	#Plot rolling statistics:
	orig = plt.plot(timeseries, color='blue',label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label = 'Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show(block=False)
	
	#Perform Dickey-Fuller test:
	print ('Results of Dickey-Fuller Test:')
	dftest = adfuller(timeseries, maxlag=0,autolag=None) #autolag='AIC'
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print (dfoutput)
	
def rmse_cv(model,x_train, y_train,cv=5):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = 5))
	#with scoring being blank, by default this would've outputted the accuracy, ex: 95%
	#with scoring="neg_mean_squared_error", we get accuracy -1, so shows by how much you were off and it's negative
	#then with the - in front, gives you the error, but positive. 
    return(rmse)
	
	
	
def prep_hockey_data(hockey_data):

	
	hockey_data.columns=['Away','Home']

	hockey_data["MtlGoals"]=hockey_data.Home
	hockey_data["OppGoals"]=hockey_data.Home

	#separating the goals from the name
	away_goals=hockey_data.Away.str.extract('(\d+)').astype(int)
	home_goals=hockey_data.Home.str.extract('(\d+)').astype(int)

	#this version will cause the 'A value is trying to be set on a copy of a slice from a DataFrame.' problem
	#hockey_data[hockey_data.Away.str.contains("MTL")]."MtlGoals"=away_goals[hockey_data.Away.str.contains("MTL")].values

	mtl_home=hockey_data.Home.str.contains("MTL")
	mtl_away=~hockey_data.Home.str.contains("MTL")

	hockey_data.loc[mtl_away,"MtlGoals"]=away_goals[mtl_away].values
	hockey_data.loc[mtl_home,"MtlGoals"]=home_goals[mtl_home].values

	hockey_data.loc[mtl_home,"OppGoals"]=away_goals[mtl_home].values
	hockey_data.loc[mtl_away,"OppGoals"]=home_goals[mtl_away].values

	hockey_data.Away=hockey_data.Away.str.replace('\d+', '')
	hockey_data.Home=hockey_data.Home.str.replace('\d+', '')

	hockey_data["Opp"]=hockey_data.Away
	hockey_data.loc[mtl_away,"Opp"]=hockey_data.Home[mtl_away].values
	hockey_data.loc[mtl_home,"Opp"]=hockey_data.Away[mtl_home].values

	#I need these to be numbers because later on I will be summing them up

	hockey_data.loc[mtl_away,"Away"]=1
	hockey_data.loc[mtl_away,"Home"]=0

	hockey_data.loc[mtl_home,"Away"]=0
	hockey_data.loc[mtl_home,"Home"]=1

	hockey_data["Win"]=0
	hockey_data["Tie"]=0
	hockey_data["Defeat"]=0


	wins=hockey_data.MtlGoals>hockey_data.OppGoals
	ties=hockey_data.MtlGoals==hockey_data.OppGoals
	losses=hockey_data.MtlGoals<hockey_data.OppGoals

	hockey_data.loc[wins,"Win"]=1
	hockey_data.loc[ties,"Tie"]=1
	hockey_data.loc[losses,"Defeat"]=1

	#days of the week

	hockey_data["monday"]=hockey_data.index.dayofweek==0
	hockey_data["tuesday"]=hockey_data.index.dayofweek==1
	hockey_data["wednesday"]=hockey_data.index.dayofweek==2
	hockey_data["thursday"]=hockey_data.index.dayofweek==3
	hockey_data["friday"]=hockey_data.index.dayofweek==4
	hockey_data["saturday"]=hockey_data.index.dayofweek==5
	hockey_data["sunday"]=hockey_data.index.dayofweek==6

	hockey_data.Away=pd.to_numeric(hockey_data.Away);
	hockey_data.Home=pd.to_numeric(hockey_data.Home);
	hockey_data.MtlGoals=pd.to_numeric(hockey_data.MtlGoals);
	hockey_data.OppGoals=pd.to_numeric(hockey_data.OppGoals);

	hockey_data=hockey_data.sort_index()

	monthly_hockey_data=hockey_data.resample("M").sum();
	monthly_hockey_data=monthly_hockey_data.dropna()
	return monthly_hockey_data


def time_series_info_on_y(y_serie):
	
	#lag_plot(pd.DataFrame(y_serie)) #doesn't show properly

	autocorrelation_plot(y_serie)
	
	plot_acf(y_serie, lags=31)
	plot_pacf(y_serie, lags=31)

def time_series_heatmap(y_series,lags=12):
	values=pd.DataFrame(y_series)
	number_of_lags_to_check=lags
	column_names=["t"]
	for i in list(range(1,number_of_lags_to_check+1)):
		values = pd.concat([values,values.iloc[:,-1].shift(i)], axis=1)
		column_names.append("t+%i"%i)
	
	values.columns=column_names	
	values=values.dropna()
	sns.heatmap(values.corr().abs())	
	

def ar_predict(train,test):
	# train autoregression
	model = AR(train)
	model_fit = model.fit()
	window = model_fit.k_ar # here I am saying, by window will be whatever this parameter says
	coef = model_fit.params
	print("window is")
	print(window)
	print("coefficients are")
	print(coef)
	
	# walk forward over time steps in test
	history = train[-window:]
	history = [history[i] for i in range(len(history))]
	predictions = list()
	for t in range(len(test)):
		length = len(history)
		lag = [history[i] for i in range(length-window,length)]
		yhat = coef[0]
		for d in range(window):
			yhat += coef[d+1] * lag[window-d-1]
		obs = test[t]
		predictions.append(yhat)
		history.append(obs)
		#print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	# plot
	t=test.index
	print("MEAN SQUARED ERROR OF %.3f"%error)
	plt.plot(t, test, t, predictions)
	plt.show()
	return predictions
	
def train_test(train_pct,x=None,y=None):
	#put in either x or y or both, as long as you identify them. Also put in your training pct, and I'll out the results in dictionary.
	
	data_dict={}
	if type(y) not in [pd.core.series.Series, pd.core.frame.DataFrame]:
		print("got here")
		train_size=int(x.shape[0]*train_pct)
		x_train=x[:train_size]
		x_test=x[train_size:]
		data_dict["x_train"]=x_train
		data_dict["x_test"]=x_test   
		mid_data_index=int(x_test.shape[0]*0.5)		
	if type(x) not in [pd.core.series.Series, pd.core.frame.DataFrame]:
		train_size=int(y.shape[0]*train_pct)
		y_train=y[:train_size]
		y_test=y[train_size:]   
		mid_data_index=int(y_test.shape[0]*0.5)
		data_dict["y_train"]=y_train
		data_dict["y_test"]=y_test   
	else:
		if x.shape[0]!=y.shape[0]:
			raise Exception("x and y don't have the same number of rows")
		train_size=int(y.shape[0]*train_pct)
		data=pd.concat([x,y],axis=1)
		x_train=x[:train_size]
		y_train=y[:train_size]
		x_test=x[train_size:]
		y_test=y[train_size:]
		data_dict["x_train"]=x_train
		data_dict["x_test"]=x_test				   
		data_dict["y_train"]=y_train
		data_dict["y_test"]=y_test	 
		mid_data_index=int(y_test.shape[0]*0.5)
	data_dict["mid_data_index"]=mid_data_index
	data_dict["train_test_index"]=train_size
	
	return data_dict

	
def analyze_sarimax(y,x=None,pct=0.8,season_length=12):
	# Define the p, d and q parameters to take any value between 0 and 2
	p = d = q = range(0, 2)

	# Generate all different combinations of p, q and q triplets
	pdq = list(itertools.product(p, d, q))

	# Generate all different combinations of seasonal p, q and q triplets
	seasonal_pdq = [(x[0], x[1], x[2], season_length) for x in list(itertools.product(p, d, q))]

	print('Examples of parameter combinations for Seasonal ARIMA...')
	print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
	print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
	print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
	print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


	warnings.filterwarnings("ignore") # specify to ignore warning messages
	best_score, best_cfg = float("inf"), None
	for param in pdq:
		for param_seasonal in seasonal_pdq:
			
			try:
				mod = sm.tsa.statespace.SARIMAX(y,
												exog =x,
												order=param,
												seasonal_order=param_seasonal,
												enforce_stationarity=False,
												enforce_invertibility=False)

				results = mod.fit()

				
				if results.aic< best_score:
					best_score=results.aic
					best_cfg=[param, param_seasonal]
					print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
			except:
				continue

	print("BEST FIT")			
	print('ARIMA{}x{}12 - AIC:{}'.format(best_cfg[0], best_cfg[1], best_score))
	#showing the best one
	mod = sm.tsa.statespace.SARIMAX(y,
									exog =x,
									order=best_cfg[0],#order=(1, 0, 1),
									seasonal_order=best_cfg[1],#seasonal_order=(1, 0, 1, 12),
									enforce_stationarity=False,
									enforce_invertibility=False)

	results = mod.fit()
	print(results.summary().tables[1])
	
	#showing the diagnostics
	results.plot_diagnostics(figsize=(15, 12))
	plt.show()
	
	
	data_dict=train_test(pct,x=x,y=y)
	
	y_test=data_dict["y_test"]
	#x_test=data_dict["x_test"]
	mid_data_point=data_dict["mid_data_point"]
	
	#forecasting  NOT DYNAMIC
	print("FORECASTING NOT DYNAMIC")
	pred = results.get_prediction(start=pd.to_datetime(y_test.index[0]), dynamic=False)
	#The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.
	pred_ci = pred.conf_int()
	
	ax = y.plot(label='observed')
	pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

	ax.fill_between(pred_ci.index,
					pred_ci.iloc[:, 0],
					pred_ci.iloc[:, 1], color='k', alpha=.2)

	ax.set_xlabel('Date')
	ax.set_ylabel('Y')
	plt.legend()

	plt.show()
	
	y_forecasted = pred.predicted_mean #you'd add whatever other coefficients you want here. 
	y_truth = logged_bar_data[mid_data_point:]

	# Compute the mean square error
	mse = ((y_forecasted - y_truth) ** 2).mean()
	print('The Mean Squared Error of our forecasts is {}'.format(mse))
	
	
	#forecasting DYNAMIC
	
	print("FORECASTING DYNAMIC")
	pred_dynamic = results.get_prediction(start=pd.to_datetime(y_test.index[0]), dynamic=True, full_results=True)
	pred_dynamic_ci = pred_dynamic.conf_int()
	ax = y[mid_data_point:].plot(label='observed', figsize=(20, 15))
	pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

	ax.fill_between(pred_dynamic_ci.index,
					pred_dynamic_ci.iloc[:, 0],
					pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

	ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(mid_data_point), logged_bar_data.index[-1],
					 alpha=.1, zorder=-1)

	ax.set_xlabel('Date')
	ax.set_ylabel('Y')

	plt.legend()
	plt.show()

#list_of_lags=list(range(1,5)
def make_lags(list_of_lags,y_serie):
    data=pd.DataFrame()
    for each in list_of_lags:
        lag=y_serie.shift(each)
        lag=lag.rename("lag %i"%each)
        data=pd.concat([data,lag],axis=1)
    data.dropna(inplace=True)
    return data
	
	
#This is to put it back into its original measurement.
def undo_differencing(original_y,prediction):
    original_y=pd.DataFrame(original_y)
    prediction=pd.DataFrame(prediction)
    answer = pd.Series(original_y.iloc[0].values, index=original_y.index)   

    combined=pd.concat([answer,prediction.cumsum()],axis=1).fillna(0)

    combined=combined.sum(axis=1)

    return combined

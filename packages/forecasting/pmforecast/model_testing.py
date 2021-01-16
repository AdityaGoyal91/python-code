import warnings
warnings.filterwarnings('ignore',)

import pandas as pd
import numpy as np
import math

from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.dates import AutoDateLocator
import matplotlib.ticker as ticker

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

class Forecast:
    '''
    parameters:
           df: dataframe with ds, y, and any regressors
           metric_name: string with name of metric being modeled. Will appear on charts.
           holidays: dataframe with columns holiday and date
           period length: length of each forecasting period
           num_periods: number of forecasting periods to iterate on
           interval_width: Confidence level of the error interval
           seasonality(True/False): include weely and yearly seasonality in the model. Defaults to True.
           seasonlity_mode: sets weekly and yearly seasonlity to additive or multiplicative. Defaults to additive.
           previous_model: dataframe containing the forecasted values from Prophet model. Has to contain at least the columns ds, y, and yhat
    '''

    def __init__(self,
                 df,
                 metric_name,
                 holidays,
                 period_length,
                 num_periods,
                 interval_width=.95,
                 weekly_seasonality=True,
                 monthly_seasonality=False,
                 yearly_seasonality=True,
                 seasonality_mode='additive',
                 previous_model=None
                 ):


        self.df=df
        #self.df=self.df.sort_values(by=['ds'], ascending=True, ignore_index=True)

        self.previous_model=previous_model


        if self.previous_model is not None:
            if self.previous_model['ds'].min() <= df['ds'].min():
                self.previous_model = self.previous_model.loc[self.previous_model['ds']>=df['ds'].min()]
                self.previous_model.reset_index(inplace=True, drop=True)
            else:
                self.df = df.loc[df['ds']>=self.previous_model['ds'].min()]
                self.df.reset_index(inplace=True, drop=True)
                self.previous_model.reset_index(inplace=True, drop=True)


        self.metric_name=metric_name
        self.holidays=holidays
        self.period_length=period_length
        self.num_periods=num_periods
        self.interval_width=interval_width
        self.weekly_seasonality=weekly_seasonality
        self.monthly_seasonality=monthly_seasonality
        self.yearly_seasonality=yearly_seasonality
        self.seasonality_mode=seasonality_mode

        self.regressors = self.df.columns.drop(['ds', 'y'])

        self.y_yhat = None   # dataframe to hold predicted values from the forecast loop
        self.y_yhat_hist = None    # dataframe to hold all historical values and predicted values
        self.rmse = None
        self.final_model = None   # prophet modle trained using all historical data




    # Function to predict a future time frame.
    # Must input a dataframe with future values for all regressors
    #def predict_future(self, time, future_df):
    #    future_periods = self.final_model.make_future_dataframe(periods=time)
    #    forecast = mod.predict(future)
    #    return forecast




    # main function of class. Iterates through num_periods number of models and makes predictions of length period_length
    def forecast_loop(self):

        # iterate through training and forecast loop to judge performance of model
        for i in range(0, self.num_periods):

            # set training data
            train = self.df.iloc[0:(self.df.shape[0]-self.period_length*(self.num_periods-i))]
            #test = self.df.iloc[(self.df.shape[0]-self.period_length*(self.num_periods-i)):(self.df.shape[0]-self.period_length*(self.num_periods-i+1))]
            test=self.df


            #add in previous model if exists
            if self.previous_model is not None:
                max_training_ds = train['ds'].max()
                min_training_ds = train['ds'].min()
                #print(min_training_ds)
                #print(max_training_ds)

                prev_model_yhat_subset = self.previous_model.loc[(self.previous_model['ds']>max_training_ds)][['yhat']]
                prev_model_y_subset = self.previous_model.loc[(self.previous_model['ds']>=min_training_ds) & (self.previous_model['ds']<=max_training_ds)][['y']]
                #print(prev_model_yhat_subset)
                prev_model_y_subset.columns=['prev_mod']
                prev_model_yhat_subset.columns=['prev_mod']


                prev_model_yyhat = pd.concat([prev_model_y_subset, prev_model_yhat_subset], axis=0, ignore_index=False)
                prev_model_yyhat.reset_index(inplace=True, drop=True)
                prev_model_y_subset.reset_index(inplace=True, drop=True)

                train = pd.concat([train, prev_model_y_subset], axis=1)
                test = pd.concat([test, prev_model_yyhat], axis=1)



            #create and run the model
            forecasted = self.run_prophet_model(train, test)

            # code to name to model
            if self.regressors.shape[0] > 0:
                regressor_name = self.regressors[self.regressors.shape[0]-1]
            else:
                regressor_name = 'time'

            name = 'forecast_' + regressor_name + '_' + str(self.period_length) + '_' + str(i)



            if self.regressors.shape[0] > 0:
                forecast_regressors = forecasted[list(self.regressors.values)]
            else:
                forecast_regressors = None

            # Get subset of returned dataframe from Prophet.predict
            forecast_all = forecasted[['ds', 'yhat', 'yhat_upper',
                                      'yhat_lower', 'trend']]
            if self.weekly_seasonality is True:
                forecast_all = pd.concat([forecast_all, forecasted['weekly']], axis=1)

            if self.yearly_seasonality is True:
                forecast_all = pd.concat([forecast_all, forecasted['yearly']], axis=1)

            if self.monthly_seasonality is True:
                forecast_all = pd.concat([forecast_all, forecasted['monthly']], axis=1)

            if self.holidays is not None:
                forecast_all = pd.concat([forecast_all, forecasted['holidays']], axis=1)

            forecast_all=pd.concat([forecast_all, forecast_regressors, self.df.loc[self.df['ds']<=(train['ds'].max() + timedelta(days=self.period_length))]['y']], axis=1, ignore_index=False)
            #self.all_iterations[name] = forecast_all

            forecasted_time_period = forecasted.iloc[(self.df.shape[0]-self.period_length*(self.num_periods-i)):(self.df.shape[0]-self.period_length*(self.num_periods-i-1))]
            yhat = forecasted.iloc[(self.df.shape[0]-self.period_length*(self.num_periods-i)):(self.df.shape[0]-self.period_length*(self.num_periods-i-1))]['yhat']
            y = self.df.iloc[(self.df.shape[0]-self.period_length*(self.num_periods-i)):(self.df.shape[0]-self.period_length*(self.num_periods-i-1))]['y']
            #rmse_i = self.get_rmse(a=y, b=yhat, n=self.period_length)
            rmse_i = math.sqrt(((y-yhat)**2).sum()/self.period_length)




            # add all historical data to dataframe
            if i==0:
                self.y_yhat_hist=pd.concat([forecasted.loc[forecasted['ds']<=train['ds'].max()],
                                            pd.DataFrame(self.df.loc[self.df['ds']<=train['ds'].max()]['y'])],
                                            axis=1, ignore_index=False)

            # add y, yhat, upper,  to dataframe
            forecasted_time_period=pd.DataFrame(forecasted_time_period)
            y=pd.DataFrame(y)
            y_yhat_stage=pd.concat([forecasted_time_period, y], axis=1)
            self.y_yhat=pd.concat([self.y_yhat,y_yhat_stage], axis=0, ignore_index=False)
            self.y_yhat_hist=pd.concat([self.y_yhat_hist,y_yhat_stage], axis=0, ignore_index=False)
            self.y_yhat_hist.where((pd.notnull(self.y_yhat_hist)), None)

            #add rmse to dataframe
            appended_rmse = pd.DataFrame({'ds':[self.df.iloc[(self.df.shape[0]-self.period_length*(self.num_periods-i))]['ds']],
                                          'rmse': [rmse_i]})
            self.rmse=pd.concat([self.rmse, appended_rmse], axis=0, ignore_index=True)

        #train model up to max date in dataframe and set to variable self.final_model
        self.set_final_model()

        # add prev_mod to regressor list if exists
        if self.previous_model is not None:
            self.regressors=self.regressors.insert(0, 'prev_mod')




    def run_prophet_model(self, train, test):

        mod = Prophet(interval_width=self.interval_width,
                      holidays=self.holidays,
                      yearly_seasonality=self.yearly_seasonality,
                      weekly_seasonality=self.weekly_seasonality,
                      seasonality_mode=self.seasonality_mode)

        if self.regressors.shape[0] > 0:
            for r in range(0, self.regressors.shape[0]):
                mod.add_regressor(self.regressors[r], mode='additive')

        if self.monthly_seasonality is True:
            #mod.add_seasonality(name='monthly', period=30.4, fourier_order=8)
            train['monthly'] = train.ds.dt.day
            mod.add_regressor('monthly', mode=self.seasonality_mode)
            test['monthly'] = test.ds.dt.day

        #add in previous model if exists
        if self.previous_model is not None:
            #display('adding prev model to mod')
            mod.add_regressor('prev_mod')



        mod.fit(train)
        future_periods = mod.make_future_dataframe(periods=self.period_length)
        #print(future_periods)

        future = test.loc[test['ds'].isin(future_periods['ds'])]
        #print(future)

        forecasted = mod.predict(future)

        return forecasted






    def set_final_model(self):

        mod = Prophet(interval_width=self.interval_width,
                      holidays=self.holidays,
                      yearly_seasonality=self.yearly_seasonality,
                      weekly_seasonality=self.weekly_seasonality,
                      seasonality_mode=self.seasonality_mode)

        if self.regressors.shape[0] > 0:
            for r in range(0, self.regressors.shape[0]):
                mod.add_regressor(self.regressors[r], mode='additive')

        if self.monthly_seasonality is True:
            #mod.add_seasonality(name='monthly', period=30.4, fourier_order=8)
            self.df['monthly']=self.df.ds.dt.day
            mod.add_regressor('monthly', mode=self.seasonality_mode)


        df_temp=self.df
        #add in previous model if exists
        if self.previous_model is not None:

            #display('adding prev model to mod')

            max_ds = self.df['ds'].max()
            min_ds = self.df['ds'].min()

            prev_model_y_subset = self.previous_model.loc[(self.previous_model['ds']>=min_ds) & (self.previous_model['ds']<=max_ds)][['y']]
            prev_model_y_subset.columns=['prev_mod']
            prev_model_y_subset.reset_index(inplace=True, drop=True)
            #print(prev_model_y_subset)
            df_temp = pd.concat([self.df, prev_model_y_subset], axis=1)
            #print(df_temp)
            mod.add_regressor('prev_mod')


        mod.fit(df_temp)
        self.final_model=mod




    def calc_rmse(self, y, yhat, n):
        #should add error if lengths of y and yhat are different

        yhat=yhat.tail(n)
        y=y.tail(n)
        return math.sqrt(((yhat-y)**2).sum()/n)

    def get_rmse(self, period_length):
        ##display(get_rmse(y_yhat_710_1['y'], y_yhat_710_1['yhat'], 7))
        ##rmse_temp=pd.DataFrame()

        ##for k, f in a.all_iterations.items():

          ##  rmse_i = self.calc_rmse(y=f.iloc[(f.shape[0]-period_length):f.shape[0]]['y'], yhat=f.iloc[(f.shape[0]-period_length):f.shape[0]]['yhat'], n=period_length)
          ##  appended_rmse = pd.DataFrame({'ds':[f.iloc[(f.shape[0]-period_length)]['ds']],
                                              ##'rmse': [rmse_i]})
          ##  rmse_temp = pd.concat([rmse_temp, appended_rmse], axis=0, ignore_index=True)

        y=self.y_yhat['y']
        yhat=self.y_yhat['yhat']
        n=period_length
        return self.calc_rmse(y=y, yhat=yhat, n=n)


    def plot_rmse(self):
        #plot rmse over time

        plt.figure(figsize=(20,7))
        plt.plot(self.rmse['ds'], self.rmse['rmse'], color='black')
        plt.xlabel('Date')
        plt.ylabel('RMSE')
        plt.title('RMSE Over Time Periods')
        plt.show()
        #display(rmse)


    def plot_time_series_components(self, lag=None):

        if lag is None:
            y_yhat=self.y_yhat
        else:
            y_yhat = self.y_yhat.tail(lag)

        plt.figure(figsize=(20,7))
        ax= plt.subplot()

        #ax.plot(df['ds'], df['yhat'], color='red', label='yhat')
        ax.plot(y_yhat['ds'], y_yhat['trend'], color='black', label='Trend')

        if self.weekly_seasonality is True:
            ax.plot(y_yhat['ds'], y_yhat['weekly']+y_yhat['trend'], color='mediumseagreen', label='Weekly')

        if self.yearly_seasonality is True:
            ax.plot(y_yhat['ds'], y_yhat['yearly']+y_yhat['trend'], color='firebrick', label='Yearly')

        for i in self.regressors:
            ax.plot(y_yhat['ds'], y_yhat[i]+y_yhat['trend'], label=str(i))

        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        ax.xaxis_date()

        ax.set_xlabel('Date')
        ax.set_ylabel(self.metric_name)

        ax.legend()
        ax.set_title(self.metric_name + ' time series')
        plt.show()



    def get_forecastdf_subset(self):
        df1 = self.y_yhat[['ds', 'yhat', 'y', 'trend', 'holidays', 'weekly', 'yearly']]
        df2 = self.y_yhat[self.regressors]

        return pd.concat([df1, df2], axis=1, ignore_index=False)

    def plot_time_series(self, lag=None, xtick_length=7):
        print('Period Length: ' + str(self.period_length))
        print('\n' + 'Num Periods: ' + str(self.num_periods))
        print('\n' + 'Regressors: ' + str(self.regressors.tolist()))


        if lag is None:
            y_yhat=self.y_yhat
        else:
            y_yhat = self.y_yhat.tail(lag)

        plt.figure(figsize=(20,7))
        ax= plt.subplot()

        ax.plot(y_yhat[['ds']], y_yhat['yhat_upper'], color='whitesmoke', label='Upper')
        ax.plot(y_yhat[['ds']], y_yhat['yhat_lower'], color='whitesmoke', label='Lower')
        ax.plot(y_yhat[['ds']], y_yhat['y'], color='black', label='Actual')
        ax.plot(y_yhat[['ds']], y_yhat['yhat'], color='red', label='Predicted')
        #ax.plot(self.y_yhat[['ds']], self.y_yhat['yhat_upper'], color='whitesmoke', label='Upper')
        #ax.plot(self.y_yhat[['ds']], self.y_yhat['yhat_lower'], color='whitesmoke', label='Lower')
        ax.fill_between(pd.to_datetime(y_yhat['ds']), y_yhat['yhat_lower'], y_yhat['yhat_upper'], facecolor='whitesmoke')
        ax.set_xlabel('Date', fontsize=20)

        #ax.xaxis.set_major_locator(AutoDateLocator())
        num_ticks = self.num_periods
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_length))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        ax.xaxis.grid()
        ax.xaxis_date()
        ax.set_ylabel(self.metric_name, fontsize=20)
        ax.tick_params(axis='both', labelsize=12)

        #ax.text(.05, .05, 'Regressors: '+ str(self.regressors.tolist()) , transform=ax.transAxes, fontsize=15)
        #+ str(self.regressors.tolist())
        ax.legend()
        ax.set_title('Predicted vs Actual', fontsize=24)
        #fig.autofmt_xdate()
        plt.show()


    def plot_outliers(self, df_outlier, lag=None, bands='sd', iqr_lower=None, iqr_upper=None, bandcolor='lightgrey', outliercolor='blue', outliersymbol = 'o', outliersize = 8):

            if lag is None:
                y_yhat = self.y_yhat_hist
            else:
                y_yhat = self.y_yhat_hist.tail(lag)

            min_date = y_yhat.ds.min()
            max_date = y_yhat.ds.max()
            # truncate outlier df to be same time frame as y_yhat_hist
            df_outlier=df_outlier[df_outlier['ds']>=min_date]
            df_outlier=df_outlier[df_outlier['ds']<=max_date]

            plt.figure(figsize=(20,7))
            ax= plt.subplot()
            if bands=='sd':
                ax.plot(y_yhat[['ds']], y_yhat['yhat_upper'], color=bandcolor, label='Upper')
                ax.plot(y_yhat[['ds']], y_yhat['yhat_lower'], color=bandcolor, label='Lower')
                ax.fill_between(pd.to_datetime(y_yhat['ds']), y_yhat['yhat_lower'], y_yhat['yhat_upper'], facecolor=bandcolor)
            if bands=='iqr' and iqr_lower is not None and iqr_upper is not None:
                ax.plot(y_yhat[['ds']], y_yhat['yhat']+iqr_upper, color=bandcolor, label='IQR Upper')
                ax.plot(y_yhat[['ds']], y_yhat['yhat']+iqr_lower, color=bandcolor, label='IQR Lower')
                ax.fill_between(pd.to_datetime(y_yhat['ds']), y_yhat['yhat']+iqr_upper, y_yhat['yhat']+iqr_lower, facecolor=bandcolor)
            ax.plot(y_yhat[['ds']], y_yhat['y'], color='black', label='Actual')
            ax.plot(y_yhat[['ds']], y_yhat['yhat'], color='red', label='Predicted')
            ax.plot(df_outlier['ds'], df_outlier['y'], outliersymbol, markersize= outliersize, color=outliercolor, label='Outliers')

            ax.set_xlabel('Date', fontsize=20)

            #ax.xaxis.set_major_locator(AutoDateLocator())
            num_ticks = self.num_periods
            ax.xaxis.set_major_locator(ticker.MultipleLocator(self.period_length))

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
            ax.xaxis.grid()
            ax.xaxis_date()
            ax.set_ylabel(self.metric_name, fontsize=20)
            ax.tick_params(axis='both', labelsize=12)

            #ax.text(.05, .05, 'Regressors: '+ str(self.regressors.tolist()) , transform=ax.transAxes, fontsize=15)
            #+ str(self.regressors.tolist())
            ax.legend()
            ax.set_title('Predicted vs Actual', fontsize=24)
            #fig.autofmt_xdate()
            plt.show()

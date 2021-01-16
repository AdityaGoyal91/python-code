
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

from .model_testing import Forecast

class Grassroots_Iteration:
    """
    Class to iterate through several different forecasting models.
    Creates a dictionary of forecast loops - one for each model. The RMSEs of each model can then be compared.
    Inputs:
        - df: Dataframe compatible with Prophet package. Must contain a minimum of two columns : ds and y. All other
              columns will be assumed to be additional regressors to interatively added to the models.
        - metric name: String format of metric to be forecasted. This will appear on all plots.
        - holidays: Dataframe compatible with Prophet package. Must contain at least two columns: holiday and ds.
        - period length: # days to forecast in each forecast loop. Default is 7 days to simulate weekly forecasts.
        - num periods: # of forecasts loops in the iteration for each model.
        - previous model: Dataframe containing the values from a Prophet model. Dataframe must contain the
                          columns ds and yhat. All other columns will be ignored. Defaults to None (no previous model).
        - split ad spend: Boolean value that specifies if ad spend by growth segment is in the model as separate
                          variables. This is a special case where all 4 variables are added at one time. This is
                          specific to models where ad spend by growth segment is a regressors. Defaults to False.
        - seasonality_mode: Can be either 'additive' or 'multiplicative'. Will set yearly and weekly seasonality as well as holiday
                            effect to multiplicative against the trend. Default is additive.
    """





    def __init__(self,
                 df,
                 metric_name,
                 holidays,
                 period_length=[7],
                 num_periods=10,
                 previous_model=None,
                 weekly_seasonality=True,
                 monthly_seasonality=False,
                 yearly_seasonality=True,
                 split_ad_spend=False,
                 seasonality_mode='additive'):

        self.df=df
        #self.df=self.df.sort_values(by=['ds'], ascending=True, ignore_index=True)

        self.metric_name=metric_name
        self.holidays=holidays
        self.period_length=period_length
        self.period_length_count=len(period_length)
        self.num_periods=num_periods
        self.weekly_seasonality=weekly_seasonality
        self.monthly_seasonality=monthly_seasonality
        self.yearly_seasonality=yearly_seasonality
        self.seasonality_mode=seasonality_mode
        self.regressors = self.df.columns.drop(['ds', 'y'])
        self.split_ad_regressors = None

        if split_ad_spend is True:
            self.regressors = self.df.columns.drop(['ds',
                                                    'y',
                                                    'mobile_ad_spend',
                                                    'transactions_ad_spend',
                                                    'other_ad_spend',
                                                    'awareness_ad_spend'
                                                    ])

            self.split_ad_regressors = self.df[['mobile_ad_spend',
                                                'transactions_ad_spend',
                                                'other_ad_spend',
                                                'awareness_ad_spend']]

        self.previous_model = previous_model

        self.split_ad_spend = split_ad_spend # note currently ad spend must be split into 4 groups:
                                             # mobile, transactions, awareness, and other

        # dictionary to save each model created
        self.all_models = {}
        self.best_model = None
        self.best_model_values = None


    # Function to set the "best" model based on minimum RMSE. Default is to calculate 28 day RMSE, but can be set to any number of days.
    # Can be run outside of the automated loop to change best model.
    def set_best_model(self, model=None, rmse_lag=28):
        if model is None:
            rmse_comp = self.compare_rmse(rmse_lag)
            name=rmse_comp.loc[rmse_comp['rmse']==rmse_comp['rmse'].min()]
            name=name.reset_index(drop=True)

            name=name.at[0, 'model']

            self.best_model = self.all_models[name].final_model
            self.best_model_values = self.all_models[name].y_yhat_hist


        else:
            self.best_model = model.final_model
            self.best_model_values = model.y_yhat_hist


    def model_iteration(self):

        regressor_count = self.regressors.shape[0]
        model_count=0

        for p in range(0, self.period_length_count):

            #go through at least one model (no regressors, seasonality only)
            df_0= self.df[['ds', 'y']]

            f0 = Forecast(df=df_0,
                          metric_name=self.metric_name,
                          holidays=self.holidays,
                          period_length=self.period_length[p],
                          num_periods=self.num_periods,
                          yearly_seasonality=self.yearly_seasonality,
                          weekly_seasonality=self.weekly_seasonality,
                          monthly_seasonality=self.monthly_seasonality,
                          seasonality_mode=self.seasonality_mode)

            f0.forecast_loop()

            #create key for dictionary
            regressor_name = 'time'
            name = 'forecast_' + regressor_name + '_' + str(self.period_length[p])# + '_' + model_count
            print(name)

            #add forecast class object to dictionary
            self.all_models[name]=f0



            #iterate through the rest of the regressors - creating one forecast class for each as they are added
            for r in range(0, regressor_count):

                regressors_iter = self.regressors[0:(r+1)]
                df_iter = pd.concat([self.df[['ds', 'y']], self.df[regressors_iter]], axis=1) #new logic for split ad spend

                #df_iter = self.df.iloc[:,0:(3+r)] #need to change this to take our split ad spend

                f = Forecast(df=df_iter,
                             metric_name=self.metric_name,
                             holidays=self.holidays,
                             period_length=self.period_length[p],
                             num_periods=self.num_periods,
                             yearly_seasonality=self.yearly_seasonality,
                             weekly_seasonality=self.weekly_seasonality,
                             monthly_seasonality=self.monthly_seasonality,
                             seasonality_mode=self.seasonality_mode)
                f.forecast_loop()
                regressor_name = self.regressors[r]

                name = 'forecast_' + regressor_name + '_' + str(self.period_length[p])# + '_' + model_count
                print(name)
                self.all_models[name]=f

            #run through loop again with split ad spend
            if self.split_ad_spend is True:

                regressors_subset = self.regressors[self.regressors!='ad_spend']
                regressor_count = len(regressors_subset )

                df_iter = self.df[['ds',
                                   'y',
                                   'mobile_ad_spend',
                                   'transactions_ad_spend',
                                   'awareness_ad_spend',
                                   'other_ad_spend']]

                f = Forecast(df=df_iter,
                             metric_name=self.metric_name,
                             holidays=self.holidays,
                             period_length=self.period_length[p],
                             num_periods=self.num_periods,
                             yearly_seasonality=self.yearly_seasonality,
                             weekly_seasonality=self.weekly_seasonality,
                             monthly_seasonality=self.monthly_seasonality,
                             seasonality_mode=self.seasonality_mode)

                f.forecast_loop()
                regressor_name = 'ad_spend'

                name = 'forecast_split_' + regressor_name + '_' + str(self.period_length[p])# + '_' + model_count
                print(name)
                self.all_models[name]=f


                for r in range(0, regressor_count):

                    regressors_iter = regressors_subset[0:(r+1)]

                    df_iter = pd.concat([
                                         self.df[['ds',
                                                  'y',
                                                  'mobile_ad_spend',
                                                  'transactions_ad_spend',
                                                  'awareness_ad_spend',
                                                  'other_ad_spend']
                                         ],
                                         self.df[regressors_iter]], axis=1) #new logic for split ad spend

                    f = Forecast(df=df_iter,
                                 metric_name=self.metric_name,
                                 holidays=self.holidays,
                                 period_length=self.period_length[p],
                                 num_periods=self.num_periods,
                                 yearly_seasonality=self.yearly_seasonality,
                                 weekly_seasonality=self.weekly_seasonality,
                                 monthly_seasonality=self.monthly_seasonality,
                                 seasonality_mode=self.seasonality_mode)

                    f.forecast_loop()
                    regressor_name = regressors_subset[r]

                    name = 'forecast_split_' + regressor_name + '_' + str(self.period_length[p])# + '_' + model_count
                    print(name)
                    self.all_models[name]=f

            if self.previous_model is not None:
                self.add_prev_model(p)

            self.set_best_model()

    def add_prev_model(self, p):
        #add logic for checking the date

        previous_model = self.previous_model.copy()
        prev_mod_min_date = previous_model['ds'].min()
        prev_mod_max_date = previous_model['ds'].max()

        # get regressors that are in the best model and create dataframe
        rmse_comp = self.compare_rmse(28)


        name=rmse_comp.loc[rmse_comp['rmse']==rmse_comp['rmse'].min()]#['model']

        name=name.reset_index(drop=True)

        name=name.at[0,'model']


        best_model = self.all_models[name]
        mod_regressors=best_model.regressors


        df_iter = pd.concat([
                             self.df[['ds', 'y']],
                             self.df[mod_regressors]
                            ],
                            axis=1)

        #make sure date range for both models is the same
        best_model_min_date = best_model.y_yhat_hist['ds'].min()
        prev_mod_min_date = previous_model['ds'].min()

        if prev_mod_min_date <= best_model_min_date:
            previous_model = previous_model.loc[previous_model['ds']>=best_model_min_date]
        else:
            df_iter = df_iter.loc[df_iter['ds']>=prev_mod_min_date]

        df_iter.reset_index(inplace=True, drop=True)


        f = Forecast(df=df_iter,
                     metric_name=self.metric_name,
                     holidays=self.holidays,
                     period_length=self.period_length[p],
                     num_periods=self.num_periods,
                     seasonality_mode=self.seasonality_mode,
                     previous_model=previous_model)

        f.forecast_loop()
        regressor_name = 'prev_mod'

        name = 'forecast_split_' + regressor_name + '_' + str(self.period_length[p])# + '_' + model_count
        print(name)
        self.all_models[name]=f


    def plot_all_time_series(self):
        for k, i in self.all_models.items():
             i.plot_time_series()


    def compare_rmse(self, lag=28):
        rmse_all_models = pd.DataFrame()

        for k, i in self.all_models.items():

            rmse_value=  i.get_rmse(lag)

            rmse_i = pd.DataFrame({'model':[k],
                                   'rmse':[rmse_value]})

            rmse_all_models=pd.concat([rmse_all_models, rmse_i], axis=0)

        rmse_all_models=rmse_all_models.reset_index(drop=True)

        return rmse_all_models

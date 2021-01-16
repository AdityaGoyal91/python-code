import warnings
warnings.filterwarnings('ignore',)

import pandas as pd
import numpy as np


from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

from .model_testing import Forecast
from .model_creation import Grassroots_Iteration


# Function to predict future value using prophet forecasting model.
# Only difference between fbprophet.predict and this function is that
# this function takes in a previous model as a parameter. If prev_mod=None,
# then the functions are identical
def predict(model, df, prev_mod=None):
    pm = pd.DataFrame()
    pm['prev_mod']=prev_mod
    return model.predict(pd.concat([df, pm['prev_mod']], axis=1))

def test():
    print("forecast_testing package is working!")

from .predict import predict_future
from .date_cleaning import create_features
from .model_training import PredictionModel
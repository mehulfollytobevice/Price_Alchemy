import pytest
import numpy as np
from price_alchemy import train
from sklearn.linear_model import LinearRegression
import pandas as pd

# create a function to test the training function 
def test_training():

    # Generate synthetic data for regression
    np.random.seed(42)

    # Number of data points
    num_data_points = 1000

    # Generate random data points
    X = np.random.rand(num_data_points, 1) * 10  # Feature values
    y = 3 * X + np.random.normal(0, 1, size=(num_data_points, 1)) 
    y=abs(y)

    # define a model 
    model= LinearRegression()

    # train 
    trained_model, metrics= train.train_model(X,y, model)

    assert type(model) == type(trained_model)
    assert metrics["mse"]== pytest.approx(0.95839805)
    assert metrics["rmse"]== pytest.approx(0.9780165)
    assert metrics["r_2"]== pytest.approx(0.98741987)
    assert metrics["rmsle"]== pytest.approx(0.14085945)



    

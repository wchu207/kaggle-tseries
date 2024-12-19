from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.varmax import VARMAX

class SMWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, order=(1, 0), trend='c'):
        self.order = order
        self.trend = trend

    def set_params(self):
        pass

    def fit(self, X, exog=None):
        # Accepts sales and promotion data as Numpy arrays
        # Each row represents a date
        # Should use STL decomposition to reduce to stationarity
        #   Save trend and seasonality to add back in prediction
        self.varmax = VARMAX(X,
                               exog=exog,
                               order=self.order,
                               trend=self.trend
                              )

    def predict(self, X):
        pass
from preprocessor import Preprocessor
from varmax_wrapper import VARMAXWrapper

class ForecastingModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.varmax = VARMAXWrapper()
        self.stl_df = None
        
    def fit(self, df, date_label, target_label, exog_label):
        series = self.preprocessor.pivot(df, date_label, [target_label, exog_label])
        series = self.preprocessor.impute(series)
        self.stl_df = self.preprocessor.reduce_to_stationarity(series[target_label])

    def forecast(self, X, n):
        if self.stl_df == None:
            raise Exception("Must fit model before forecasting")
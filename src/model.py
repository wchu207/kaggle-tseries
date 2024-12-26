from preprocessor import Preprocessor
from multi_stl import MultiSTL
from varmax_wrapper import VARMAXWrapper

class ForecastingModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.multistl = MultiSTL()
        self.varmax = VARMAXWrapper()
        
    def fit(self, df, date_label, target_label, exog_label):
        series = self.preprocessor.pivot(df, date_label, [target_label, exog_label])
        series = self.preprocessor.impute(series)
        self.multistl.fit(series[target_label])

    def forecast(self, X, n):
        if self.stl_df == None:
            raise Exception("Must fit model before forecasting")
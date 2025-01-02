from preprocessor import Preprocessor
from multi_stl import MultiSTL
from varmax_wrapper import VARMAXWrapper

# Contains a collection of models, one per store
class ForecastingModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.multistl = MultiSTL()
        self.varmax = VARMAXWrapper()
        
        self.data = None
        self.start_date = None
        self.no_data = None
        self.date_label = None
        self.target_label = None
        self.exog_label = None
        
    def fit(self, df, date_label, target_label, exog_label):
        series = self.preprocessor.pivot(df, date_label, [target_label, exog_label])
        series = self.preprocessor.impute(series)
        self.multistl.fit(series[target_label])

        self.data = series
        self.start_date = series.index[0]
        self.date_label = date_label
        self.target_label = target_label
        self.exog_label = exog_label

        self.fit_varmax()

    def fit_varmax(self):
        # Eliminate series with exactly zero sales on all days
        self.no_data = (self.data["sales"] == 0).all()
        self.varmax.fit(
            self.data[self.target_label].loc[:, ~self.no_data],
            self.data[self.exog_label].loc[:, ~self.no_data]
        )

    def forecast(self, n):
        self.varmax.forecast(n)
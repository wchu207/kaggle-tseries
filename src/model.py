from preprocessor import Preprocessor
from store_model import StoreModel
from varmax_wrapper import VARMAXWrapper
from concurrent.futures import ProcessPoolExecutor

# Contains a collection of models, one per store
class ForecastingModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.varmax = VARMAXWrapper()
        
        self.data = None
        self.start_date = None
        self.no_data = None
        self.date_label = None
        self.target_label = None
        self.exog_label = None
        self.p = 1
        self.q = 0

        self.stores = None
        
    def fit(self, df, date_label, target_label, exog_label):
        series = self.preprocessor.pivot(df, date_label, [target_label, exog_label])
        series = self.preprocessor.impute(series)

        self.data = series.sort_index(axis=1)
        self.start_date = series.index[0]
        self.date_label = date_label
        self.target_label = target_label
        self.exog_label = exog_label

        self.stores = self.data.columns.get_level_values(1).unique().values
        print(self.stores)

        results = None
        with ProcessPoolExecutor(8) as f:
            results = f.map(
                self.fit_store,
                self.stores
            )
        self.models = dict(zip(self.stores, results))

    def fit_store(self, store_nbr):
        endog = self.data[(self.target_label, store_nbr)]
        exog = self.data[(self.exog_label, store_nbr)]
        return StoreModel().fit(endog, exog, self.p, self.q)
        



    def forecast(self, n):
        self.varmax.forecast(n)
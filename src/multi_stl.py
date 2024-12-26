import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
from concurrent.futures import ProcessPoolExecutor

class MultiSTL:
    def __init__(self):
        self.seasonal_7 = None
        self.seasonal_365 = None

    def reduce_to_stationarity(self, df):
        # Accepts dataframe with multi-index (store_nbr, family)
        #   Returns new dataframe with residuals, trend, seasons
        #   i.e. columns are multi-indexed (store_nbr, family, output)
        #   "onpromotion" left untouched, this is a new dataframe
        nobs = df.shape[0]
        store_nbrs = df.columns.levels[0]
        product_fams = df.columns.levels[1]
        outputs = ["resid", "trend", "seasonal_7", "seasonal_365"]
        
        output_index = pd.MultiIndex.from_product([outputs, store_nbrs, product_fams])
        output_df = pd.DataFrame(index=df.index, columns=output_index)

        results = None
        with ProcessPoolExecutor(8) as f:
            results = f.map(
                self.reduce_column,
                [df[column] for column in df.columns]
            )
        for i, result in enumerate(results):
            output_df[("resid", *df.columns[i])] = result.resid
            output_df[("trend", *df.columns[i])] = result.trend
            output_df[("seasonal_7", *df.columns[i])] = result.seasonal["seasonal_7"]
            output_df[("seasonal_365", *df.columns[i])] = result.seasonal["seasonal_365"]

        return output_df

    def fit(self, df):
        stl_df = self.reduce_to_stationarity(df)
        self.seasonal_7 = self.fit_all_seasonal_7(stl_df['seasonal_7'])
        self.seasonal_365 = self.fit_all_seasonal_365(stl_df['seasonal_365'])
        
    def reduce_column(self, series):
        results = MSTL(series, periods=[7, 365]).fit()
        return results

    def fit_trend(self, trend):
        pass

    def fit_seasonal_7(self, seasonal_7):
        return seasonal_7.tail(21).to_numpy().reshape(3, 7).mean(axis=0)
        
    def fit_all_seasonal_7(self, seasonal_7_df):
        results = None
        with ProcessPoolExecutor(8) as f:
            results = f.map(
                self.fit_seasonal_7,
                [seasonal_7_df[column] for column in seasonal_7_df.columns]
            )
        return dict(zip(seasonal_7_df.columns, results))

    def fit_seasonal_365(self, seasonal_365):
        return seasonal_365.tail(3*365).to_numpy().reshape(3, 365).mean(axis=0)
        
    def fit_all_seasonal_365(self, seasonal_365_df):
        results = None
        with ProcessPoolExecutor(8) as f:
            results = f.map(
                self.fit_seasonal_365,
                [seasonal_365_df[column] for column in seasonal_365_df.columns]
            )
        return dict(zip(seasonal_365_df.columns, results))

    def restore_season_and_trend(self, resids, stl_df):
        pass
    

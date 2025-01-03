import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
from concurrent.futures import ProcessPoolExecutor
from statsmodels.regression.linear_model import OLS

class MultiSTL:
    def __init__(self):
        self.seasonal_7 = None
        self.seasonal_365 = None
        self.trend = None
        self.resid = None

    def reduce_to_stationarity(self, df):
        # Accepts dataframe with multi-index (store_nbr, family)
        #   Returns new dataframe with residuals, trend, seasons
        #   i.e. columns are multi-indexed (store_nbr, family, output)
        #   "onpromotion" left untouched, this is a new dataframe
        nobs = df.shape[0]
        product_fams = df.columns
        outputs = ["resid", "trend", "seasonal_7", "seasonal_365"]
        
        output_index = pd.MultiIndex.from_product([outputs, product_fams])
        output_df = pd.DataFrame(index=df.index, columns=output_index)

        results = None
        with ProcessPoolExecutor(8) as f:
            results = f.map(
                self.reduce_column,
                [df[column] for column in df.columns]
            )
        for i, result in enumerate(results):
            output_df[("resid", df.columns[i])] = result.resid
            output_df[("trend", df.columns[i])] = result.trend
            output_df[("seasonal_7", df.columns[i])] = result.seasonal["seasonal_7"]
            output_df[("seasonal_365", df.columns[i])] = result.seasonal["seasonal_365"]

        return output_df

    def fit(self, df):
        stl_df = self.reduce_to_stationarity(df)
        self.seasonal_7 = self.fit_all_seasonal_7(stl_df['seasonal_7'])
        self.seasonal_365 = self.fit_all_seasonal_365(stl_df['seasonal_365'])
        self.trend = self.fit_trend(stl_df['trend'])
        self.resid = stl_df['resid']

    def get_residuals(self):
        return self.resid
        
    def reduce_column(self, series):
        results = MSTL(series, periods=[7, 365]).fit()
        return results

    def fit_trend(self, trend):
        return trend.apply(
            lambda col: OLS(
                col.values,
                sm.add_constant(col.index.values.astype(float)),
                hasconst=True).fit()
        )

        

    def fit_seasonal_7(self, seasonal_7):
        return seasonal_7.tail(21).to_numpy().reshape(3, 7).mean(axis=0)

    # Returns map of (store_nbr, family) : array of size 7
    #   Array: offset for each day of the week, starting with 
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
    

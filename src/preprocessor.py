import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller

class Preprocessor:
    def __init__(self):
        self.stl_df = None
        pass

    def pivot(self, df):
        # Accepts DataFrame with rows containing 'date', 'store_nbr', 'family', 'onpromotion', 'sales'
        # Pivots to index by date, columns are store_nbr x family, values are sales and/or onpromotion
        values = ['sales', 'onpromotion']
        values = [value for value in values if value in df.columns]
        return pd.pivot(df, index='date', columns=['store_nbr', 'family'], values=values)
    
    def impute(self, series):
        # Accepts dataframe indexed by dates, with daily data
        # Inserts rows for missing days and imputes by copying previous day's values
        dates = pd.date_range(series.index.min(), series.index.max())
        out_series = series.reindex(dates, fill_value=np.nan)
        out_series = out_series.ffill()
        return out_series

    def reduce_to_stationarity(self, df):
        # Accepts dataframe with multi-index (store_nbr, family)
        #   Returns new dataframe with residuals, trend, seasons
        #   i.e. columns are multi-indexed (store_nbr, family, output)
        #   "onpromotion" left untouched, this is a new dataframe
        nobs = df.shape[0]
        store_nbrs = df.columns.levels[0]
        product_fams = df.columns.levels[1]
        outputs = ["resid", "trend", "seasonal_7", "seasonal_365"]
        
        output_index = pd.MultiIndex.from_product([store_nbrs, product_fams, outputs])
        output_df = pd.DataFrame(index=df.index, columns=output_index)

        for column in tqdm(df.columns):
            results = self.reduce_column(df[column])
            output_df[(*column, "resid")] = results.resid
            output_df[(*column, "trend")] = results.trend
            output_df[(*column, "seasonal_7")] = results.seasonal["seasonal_7"]
            output_df[(*column, "seasonal_365")] = results.seasonal["seasonal_365"]
        
        self.stl_df = output_df
        return output_df
        
    def reduce_column(self, series):
        results = MSTL(series, periods=[7, 365]).fit()
        return results

    def restore_season_and_trend(self, df):
        pass
    
    def fit(self, X):
        pass

    def transform(self, X):
        # X: raw dataframe
        # Should return X_endog and X_exog 
        # After pivoting and imputing
        # For X_endog: also detrend and de-seasonalize
        out_df = df.copy()
        out_df['date'] = pd.to_datetime(df['date'])
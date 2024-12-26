import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
from concurrent.futures import ProcessPoolExecutor

class Preprocessor:
    def __init__(self):
        pass

    def pivot(self, df, index_label, values_list):
        # Accepts DataFrame with rows containing 'date', 'store_nbr', 'family', 'onpromotion', 'sales'
        # Pivots to index by date, columns are store_nbr x family, values are sales and/or onpromotion
        # New "columns": ['sales', 'onpromotion'] x store_nbr x family
        values = [column for column in values_list if column in df.columns]
        columns = [column for column in df.columns if column not in values and column != index_label]
        return pd.pivot(df, index=index_label, columns=columns, values=values)
    
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

        results = None
        with ProcessPoolExecutor(8) as f:
            results = f.map(
                self.reduce_column,
                [df[column] for column in df.columns]
            )
        for i, result in enumerate(results):
            output_df[(*df.columns[i], "resid")] = result.resid
            output_df[(*df.columns[i], "trend")] = result.trend
            output_df[(*df.columns[i], "seasonal_7")] = result.seasonal["seasonal_7"]
            output_df[(*df.columns[i], "seasonal_365")] = result.seasonal["seasonal_365"]

        return output_df
        
    def reduce_column(self, series):
        results = MSTL(series, periods=[7, 365]).fit()
        return results

    def restore_season_and_trend(self, resids, stl_df):
        pass
    

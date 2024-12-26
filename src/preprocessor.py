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

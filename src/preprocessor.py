class Preprocessor:
    def __init__(self):
        pass

    def pivot(self, df):
        # Accepts DataFrame with rows containing 'date', 'store_nbr', 'family', 'onpromotion', 'sales'
        # Pivots to index by date, columns are store_nbr x family, values are sales and/or onpromotion
        values = ['sales', 'onpromotion']
        values = [value in values if value in df.columns]
        return pd.pivot(new_df, index='date', columns=['store_nbr', 'family'], values=values)
    
    def impute(self, df):
        # Accepts dataframe indexed by dates, with daily data
        # Inserts rows for missing days and imputes by copying previous day's values
        pass

    def fit(self, X):
        pass

    def transform(self, X)
        # X: raw dataframe
        # Should return X_endog and X_exog as Numpy arrays
        # After pivoting and imputing
        out_df = df.copy()
        out_df['date'] = pd.to_datetime(df['date'])
from multi_stl import MultiSTL
from varmax_wrapper import VARMAXWrapper

class StoreModel:
    def __init__(self):
        self.stl = MultiSTL()
        self.varmax = VARMAXWrapper()
        self.endog = None
        self.exog = None
        self.no_data = None

    def fit(self, endog_df, exog_df, p, q):
        # endog_df: for a single store, columns are indexed by family
        # exog_df: like endog_df
        self.stl.fit(endog_df)
        self.endog = self.stl.get_residuals()
        self.exog = exog_df
        
        # Eliminate series with exactly zero sales on all days
        self.no_data = (self.endog == 0).all()
        self.varmax.fit(
            self.endog.loc[:, ~self.no_data],
            self.exog.loc[:, ~self.no_data]
        )
        


    def forecast(self, n):
        pass
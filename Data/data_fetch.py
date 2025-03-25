# We use this class to read the data from the csv file

import pandas as pd

class DataFetch:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.pricing = None
        self.returns = None
        self.tickers_list = None
        self.reading_processing_data()

    def reading_processing_data(self):
        data = pd.read_csv(self.file_name, index_col=0)
        data = data.apply(pd.to_numeric, errors='coerce')
        self.returns = data.pct_change().dropna()
        self.pricing = data
        self.tickers_list = list(self.returns.columns.unique())

    def get_pricing(self):
        return self.pricing

    def get_returns(self):
        return self.returns

    def get_tickers_list(self):
        return self.tickers_list
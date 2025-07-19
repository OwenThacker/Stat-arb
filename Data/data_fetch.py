import pandas as pd

class DataFetch:
    """
    A class to load and process financial time series data from a CSV file.

    This class reads pricing data, converts the index to datetime,
    ensures numeric formatting, calculates returns, and stores
    tickers for further use in trading or analysis.

    Attributes
    ----------
    file_name : str
        Path to the CSV file containing pricing data.
    pricing : pd.DataFrame
        DataFrame holding the raw pricing data after processing.
    returns : pd.DataFrame
        DataFrame holding percentage returns computed from pricing data.
    tickers_list : list
        List of unique tickers found in the dataset.
    """

    def __init__(self, file_name: str):
        """
        Initializes the DataFetch object and processes the data.

        Parameters
        ----------
        file_name : str
            The path to the CSV file to read.
        """
        self.file_name = file_name
        self.pricing = None
        self.returns = None
        self.tickers_list = None
        self.reading_processing_data()

    def reading_processing_data(self):
        """
        Reads data from the CSV file and processes it:
        - Converts index to datetime
        - Converts all values to numeric (coercing errors to NaN)
        - Calculates percentage returns
        - Stores pricing and tickers list
        """
        data = pd.read_csv(self.file_name, index_col=0)
        data.index = pd.to_datetime(data.index, errors='coerce')
        data = data.apply(pd.to_numeric, errors='coerce')
        self.returns = data.pct_change().dropna()
        self.pricing = data
        self.tickers_list = list(self.returns.columns.unique())

    def get_pricing(self) -> pd.DataFrame:
        """
        Returns the processed pricing data.

        Returns
        -------
        pd.DataFrame
            The pricing data with datetime index.
        """
        return self.pricing

    def get_returns(self) -> pd.DataFrame:
        """
        Returns the computed returns data.

        Returns
        -------
        pd.DataFrame
            Percentage change returns for each ticker.
        """
        return self.returns

    def get_tickers_list(self) -> list:
        """
        Returns the list of tickers from the dataset.

        Returns
        -------
        list
            List of ticker symbols present in the CSV file.
        """
        return self.tickers_list

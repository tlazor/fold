import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TsvToDataFrame(BaseEstimator, TransformerMixin):
    """
    A custom transformer that reads a TSV (tab-separated) file and returns a pandas DataFrame.
    """
    def __init__(self, file_path, encoding='utf-8', nrows=None):
        """
        Parameters
        ----------
        file_path : str
            Path to the TSV file.
        encoding : str
            File encoding (default is 'utf-8').
        """
        self.file_path = file_path
        self.encoding = encoding
        self.nrows = nrows

    def fit(self, X=None, y=None):
        """
        This transformer doesn't learn anything from the data, so fit does nothing.
        """
        return self

    def transform(self, X=None):
        """
        Reads the TSV file specified by self.file_path and returns it as a pandas DataFrame.
        """
        df = pd.read_csv(self.file_path, sep='\t', encoding=self.encoding, nrows=self.nrows)
        return df
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Un6Transformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that reads aligned UN corpus files and returns a pandas DataFrame.
    Files are expected to be named like 'UNv1.0.6way.en', 'UNv1.0.6way.ar', etc.
    """

    def __init__(self, file_path, langs, encoding="utf-8", nrows=None):
        """
        Parameters
        ----------
        file_path : str or Path
            Path to the directory containing the UN corpus files.
        langs : list
            List of language codes to include (e.g., ['en', 'ar', 'zh', 'fr', 'ru', 'es']).
        encoding : str
            File encoding (default is 'utf-8').
        nrows : int, optional
            Number of rows to read (default is None, meaning read all rows).
        """
        self.file_path = file_path
        self.langs = langs
        self.encoding = encoding
        self.nrows = nrows

    def fit(self, X=None, y=None):
        """
        This transformer doesn't learn anything from the data, so fit does nothing.
        """
        return self

    def transform(self, X=None):
        """
        Reads the aligned UN corpus files and returns them as a pandas DataFrame.
        Each column corresponds to one language.
        """
        # Construct filenames for each language
        filenames = [f"UNv1.0.6way.{lang}" for lang in self.langs]
        
        # Open all files
        files = [
            open(self.file_path / filename, "r", encoding=self.encoding)
            for filename in filenames
        ]

        # Read aligned lines
        aligned_lines = []
        for line_tuple in zip(*files):
            # Strip whitespace from each line
            row = [line.strip() for line in line_tuple]
            aligned_lines.append(row)
            
            # Stop if we've reached nrows
            if self.nrows and len(aligned_lines) >= self.nrows:
                break

        # Close all files
        for f in files:
            f.close()

        # Create DataFrame with language codes as column names
        df = pd.DataFrame(aligned_lines, columns=self.langs)

        return df 
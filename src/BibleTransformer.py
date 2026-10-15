import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import languages


class BibleTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that reads a TSV (tab-separated) file and returns a pandas DataFrame.
    """

    def __init__(self, file_path, langs, encoding="utf-8", nrows=None, books=66):
        """
        Parameters
        ----------
        file_path : str
            Path to the TSV file.
        encoding : str
            File encoding (default is 'utf-8').
        """
        self.file_path = file_path
        self.langs = langs
        self.encoding = encoding
        self.nrows = nrows
        self.books = books

    def fit(self, X=None, y=None):
        """
        This transformer doesn't learn anything from the data, so fit does nothing.
        """
        return self

    def transform(self, X=None):
        """
        Reads the TSV file specified by self.file_path and returns it as a pandas DataFrame.
        """
        # Fail rather than skip a language: downstream steps identify languages
        # by column position, so a dropped column would shift every label.
        bible_files = languages.source_ids("bible_file", self.langs)
        missing = [lang for lang in self.langs if lang not in bible_files]
        if missing:
            raise ValueError(f"No Bible file registered for {missing}")
        lang_filenames = [f"{bible_files[lang]}.txt" for lang in self.langs]

        aligned_lines = []
        for book_num in range(1, self.books):
            book_path = list(self.file_path.rglob(f"{book_num}-b.*"))[0]
            lang_files = [
                open(book_path / lang_fn, "r", encoding=self.encoding)
                for lang_fn in lang_filenames
            ]

            for line_tuple in zip(*lang_files):
                # line_tuple might look like ("line1 from file1\n", "line1 from file2\n", ...)
                row = [line.strip() for line in line_tuple]
                aligned_lines.append(row)

            # 5) Close the files
            for f in lang_files:
                f.close()

        # Each column corresponds to one language; each row is the x-th line from each file
        df = pd.DataFrame(aligned_lines, columns=self.langs)

        return df

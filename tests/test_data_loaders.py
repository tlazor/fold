from pathlib import Path

import pytest

from TsvToDataFrame import TsvToDataFrame
from BibleTransformer import BibleTransformer
from Un6Transformer import Un6Transformer

# ===========================================================================
# TsvToDataFrame
# ===========================================================================


class TestTsvToDataFrame:
    def test_shape(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text("col1\tcol2\tcol3\na\tb\tc\n1\t2\t3\n", encoding="utf-8")
        df = TsvToDataFrame(file_path=tsv).transform(None)
        assert df.shape == (2, 3)

    def test_column_names(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text("alpha\tbeta\tgamma\n1\t2\t3\n", encoding="utf-8")
        df = TsvToDataFrame(file_path=tsv).transform(None)
        assert list(df.columns) == ["alpha", "beta", "gamma"]

    def test_nrows_truncation(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        lines = ["a\tb"] + [f"{i}\t{i}" for i in range(10)]
        tsv.write_text("\n".join(lines), encoding="utf-8")
        df = TsvToDataFrame(file_path=tsv, nrows=3).transform(None)
        assert df.shape[0] == 3

    def test_fit_returns_self(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text("a\tb\n1\t2\n", encoding="utf-8")
        t = TsvToDataFrame(file_path=tsv)
        assert t.fit(None) is t

    def test_langs_select_and_order_columns(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text("en\tfr\tde\n1\t2\t3\n", encoding="utf-8")
        df = TsvToDataFrame(file_path=tsv, langs=["de", "en"]).transform(None)
        assert list(df.columns) == ["de", "en"]
        assert df.iloc[0].tolist() == [3, 1]

    def test_missing_lang_raises(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text("en\tfr\n1\t2\n", encoding="utf-8")
        with pytest.raises(KeyError):
            TsvToDataFrame(file_path=tsv, langs=["en", "zh"]).transform(None)


# ===========================================================================
# BibleTransformer
# ===========================================================================


class TestBibleTransformer:
    def _make_book(self, root: Path, book_name: str, lines: list, name="English"):
        d = root / book_name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

    def test_columns_named_by_lang_in_requested_order(self, tmp_path):
        self._make_book(tmp_path, "1-b.GEN", ["in the beginning"])
        self._make_book(tmp_path, "1-b.GEN", ["au commencement"], name="French")
        df = BibleTransformer(
            file_path=tmp_path, langs=["fr", "en"], books=2
        ).transform(None)
        assert list(df.columns) == ["fr", "en"]
        assert df.loc[0, "fr"] == "au commencement"

    def test_filename_comes_from_registry(self, tmp_path):
        # Persian's Bible file is Farsi.txt, not langcodes' "Persian".
        self._make_book(tmp_path, "1-b.GEN", ["x"], name="Farsi")
        df = BibleTransformer(file_path=tmp_path, langs=["fa"], books=2).transform(None)
        assert list(df.columns) == ["fa"]

    def test_missing_file_raises(self, tmp_path):
        self._make_book(tmp_path, "1-b.GEN", ["x"])
        with pytest.raises(FileNotFoundError):
            BibleTransformer(file_path=tmp_path, langs=["en", "fr"], books=2).transform(
                None
            )

    def test_lang_without_bible_file_raises(self, tmp_path):
        # Kazakh is in the registry but has no Bible translation.
        self._make_book(tmp_path, "1-b.GEN", ["x"])
        with pytest.raises(ValueError):
            BibleTransformer(file_path=tmp_path, langs=["en", "kk"], books=2).transform(
                None
            )

    def test_single_book_shape(self, tmp_path):
        self._make_book(tmp_path, "1-b.GEN", ["line1", "line2", "line3"])
        # books=2 → range(1, 2) → reads only book 1
        df = BibleTransformer(file_path=tmp_path, langs=["en"], books=2).transform(None)
        assert df.shape == (3, 1)

    def test_two_books_accumulate_rows(self, tmp_path):
        self._make_book(tmp_path, "1-b.GEN", ["a", "b"])
        self._make_book(tmp_path, "2-b.EXO", ["c", "d", "e"])
        # books=3 → range(1, 3) → reads books 1 and 2: 2 + 3 = 5 rows
        df = BibleTransformer(file_path=tmp_path, langs=["en"], books=3).transform(None)
        assert df.shape == (5, 1)

    def test_fit_returns_self(self, tmp_path):
        self._make_book(tmp_path, "1-b.GEN", ["x"])
        t = BibleTransformer(file_path=tmp_path, langs=["en"], books=2)
        assert t.fit(None) is t


# ===========================================================================
# Un6Transformer
# ===========================================================================


class TestUn6Transformer:
    def _write(self, root: Path, lang: str, lines: list):
        (root / f"UNv1.0.6way.{lang}").write_text("\n".join(lines), encoding="utf-8")

    def test_shape(self, tmp_path):
        for lang in ["en", "fr"]:
            self._write(tmp_path, lang, ["line1", "line2", "line3"])
        df = Un6Transformer(file_path=tmp_path, langs=["en", "fr"]).transform(None)
        assert df.shape == (3, 2)

    def test_column_names(self, tmp_path):
        langs = ["ar", "en", "zh"]
        for lang in langs:
            self._write(tmp_path, lang, ["a", "b"])
        df = Un6Transformer(file_path=tmp_path, langs=langs).transform(None)
        assert list(df.columns) == langs

    def test_nrows_truncation(self, tmp_path):
        for lang in ["en", "fr"]:
            self._write(tmp_path, lang, [str(i) for i in range(10)])
        df = Un6Transformer(file_path=tmp_path, langs=["en", "fr"], nrows=4).transform(
            None
        )
        assert df.shape[0] == 4

    def test_fit_returns_self(self, tmp_path):
        self._write(tmp_path, "en", ["a"])
        t = Un6Transformer(file_path=tmp_path, langs=["en"])
        assert t.fit(None) is t

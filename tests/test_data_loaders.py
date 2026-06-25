from pathlib import Path
import pytest
from TsvToDataFrame import TsvToDataFrame
from BibleTransformer import BibleTransformer
from Un6Transformer import Un6Transformer

_BIBLE_ENGLISH = Path("data/aligned/1-b.GEN/English.txt")
_BIBLE_DATA_AVAILABLE = _BIBLE_ENGLISH.exists()


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


# ===========================================================================
# BibleTransformer
#
# BibleTransformer.transform() hard-codes a CWD-relative existence check for
# data/aligned/1-b.GEN/English.txt to decide whether to include "en".
# Tests are skipped when that file is absent (headless CI without the corpus).
# The actual reading uses tmp_path, which contains synthetic book data.
# ===========================================================================

@pytest.mark.skipif(
    not _BIBLE_DATA_AVAILABLE,
    reason="Requires data/aligned/1-b.GEN/English.txt — run from repo root with full data"
)
class TestBibleTransformer:
    def _make_book(self, root: Path, book_name: str, lines: list):
        d = root / book_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "English.txt").write_text("\n".join(lines), encoding="utf-8")

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
        df = Un6Transformer(file_path=tmp_path, langs=["en", "fr"], nrows=4).transform(None)
        assert df.shape[0] == 4

    def test_fit_returns_self(self, tmp_path):
        self._write(tmp_path, "en", ["a"])
        t = Un6Transformer(file_path=tmp_path, langs=["en"])
        assert t.fit(None) is t

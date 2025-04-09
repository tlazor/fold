import langcodes
import pandas as pd
from pandas import DataFrame

# languages in XNLI dataset
XNLI_LANGUAGES = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

# weeks to reach an integrated score of 3 (Speaking + Listening) on the Interagency Language Roundtable (ILR) scale
FSI_SCALE = {
    # Category I
    "da": 24,  # Danish
    "nl": 24,  # Dutch
    "fr": 30,  # French
    "it": 24,  # Italian
    "no": 24,  # Norwegian
    "pt": 24,  # Portuguese
    "ro": 24,  # Romanian
    "es": 30,  # Spanish
    "sv": 24,  # Swedish
    # Category II
    "de": 36,  # German
    "ht": 36,  # Haitian Creole
    "id": 36,  # Indonesian
    "ms": 36,  # Malay
    "sw": 36,  # Swahili
    # Category III
    "sq": 44,  # Albanian
    "am": 44,  # Amharic
    "hy": 44,  # Armenian
    "az": 44,  # Azerbaijani
    "bn": 44,  # Bengali
    "bg": 44,  # Bulgarian
    "my": 44,  # Burmese
    "cs": 44,  # Czech
    "fa": 44,  # Dari (Farsi)
    "et": 44,  # Estonian
    "fi": 44,  # Finnish
    "ka": 44,  # Georgian
    "el": 44,  # Greek
    "he": 44,  # Hebrew
    "hi": 44,  # Hindi
    "hu": 44,  # Hungarian
    "kk": 44,  # Kazakh
    "km": 44,  # Khmer
    "ku": 44,  # Kurdish
    "ky": 44,  # Kyrgyz
    "lo": 44,  # Lao
    "lv": 44,  # Latvian
    "lt": 44,  # Lithuanian
    "mk": 44,  # Macedonian
    "mn": 44,  # Mongolian
    "ne": 44,  # Nepali
    "pl": 44,  # Polish
    "ru": 44,  # Russian
    "sr": 44,  # Serbo-Croatian
    "sk": 44,  # Slovak
    "sl": 44,  # Slovenian
    "tg": 44,  # Tajiki
    "th": 44,  # Thai
    "tr": 44,  # Turkish
    "tk": 44,  # Turkmen
    "uk": 44,  # Ukrainian
    "ur": 44,  # Urdu
    "uz": 44,  # Uzbek
    "vi": 44,  # Vietnamese
    # Category IV
    "ar": 88,  # Arabic
    "yue": 88,  # Chinese - Cantonese
    "zh": 88,  # Chinese - Mandarin
    "ja": 88,  # Japanese
    "ko": 88,  # Korean
}

# Mutual Intelligibility results from
# https://www.tandfonline.com/doi/full/10.1080/14790718.2017.1350185
GERMANIC_INTELLIGABILITY = DataFrame(
    {
        "da": [None, 10.5, 7.9, 16.7, 62.5, 24.4],
        "du": [13.3, None, 10.3, 31.1, 13.0, 16.9],
        "en": [92.1, 94.0, None, 85.7, 89.6, 90.4],
        "de": [47.8, 75.0, 27.7, None, 37.0, 46.9],
        "sv": [56.7, 10.4, 8.3, 10.0, None, 21.4],
        "Mean": [52.5, 47.5, 13.6, 35.9, 50.5, 40.0],
    },
    index=["da", "du", "en", "de", "sv", "Mean"],
)

ROMANCE_INTELLIGABILITY = DataFrame(
    {
        "fr": [None, 46.3, 34.3, 47.1, 28.2, 39.0],
        "it": [24.2, None, 49.4, 57.7, 45.7, 44.3],
        "pt": [23.5, 33.5, None, 22.9, 37.2, 29.3],
        "ro": [11.0, 10.6, 14.7, None, 13.6, 12.5],
        "es": [31.5, 65.7, 77.4, 54.0, None, 57.2],
        "Total": [22.6, 36.6, 47.2, 44.9, 32.2, 36.7],
    },
    index=["fr", "it", "pt", "ro", "es", "Total"],
)

SLAVIC_INTELLIGABILITY = DataFrame(
    {
        "bg": [None, 19.7, 13.4, 13.7, 10.1, 18.0, 15.0],
        "hr": [29.1, None, 19.4, 14.4, 25.9, 79.4, 33.6],
        "cs": [10.6, 18.1, None, 26.6, 95.0, 18.0, 33.7],
        "pl": [7.1, 9.5, 35.4, None, 50.7, 12.8, 23.1],
        "sk": [16.0, 23.0, 92.7, 40.7, None, 18.8, 21.7],
        "sl": [20.6, 43.7, 15.7, 13.4, 15.1, None, 27.6],
        "Total": [16.7, 22.8, 35.3, 21.8, 39.4, 29.4, 24.6],
    },
    index=["bg", "hr", "cs", "pl", "sk", "sl", "Total"],
)


def get_lexical_distance():
    lex_sim = pd.read_excel("./data/lexical-distance-matrix.xlsx", skiprows=2)

    # Drop the "branch" and "language" columns
    # (Adjust the column names if they are titled differently in your Excel file)
    lex_sim = lex_sim.drop(columns=["Unnamed: 0", "Unnamed: 1"])

    # Convert 3-letter abbreviations to 2-letter using langcodes
    # We assume the column is named "abbreviation" (update as needed).
    def convert_3_to_2(letter3):
        return langcodes.Language.get(
            letter3
        ).language  # This should give the 2-letter ISO 639-1 code if available

    lex_sim["abbr_2_letter"] = lex_sim["Abreviation"].apply(convert_3_to_2)

    mapping_3_to_2 = dict(zip(lex_sim["Abreviation"], lex_sim["abbr_2_letter"]))

    # 2) Rename distance columns from 3-letter to 2-letter
    #    We'll replace any column name that appears in our mapping.
    def rename_col(col):
        return mapping_3_to_2[col] if col in mapping_3_to_2 else col

    lex_sim = lex_sim.rename(columns=rename_col)

    # 3) Set the index to the 2-letter codes instead of the 3-letter ones
    lex_sim = lex_sim.set_index("abbr_2_letter")
    lex_sim = lex_sim.drop(columns=["Abreviation"])

    return lex_sim


LEXICAL_SIMILARITY = get_lexical_distance()

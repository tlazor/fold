import math
import langcodes
import pandas as pd
from pandas import DataFrame

BERT_MULTILINGUAL_LANGS = [
    "Afrikaans",
    "Albanian",
    "Arabic",
    "Aragonese",
    "Armenian",
    "Asturian",
    "Azerbaijani",
    "Bashkir",
    "Basque",
    "Bavarian",
    "Belarusian",
    "Bengali",
    "Bishnupriya Manipuri",
    "Bosnian",
    "Breton",
    "Bulgarian",
    "Burmese",
    "Catalan",
    "Cebuano",
    "Chechen",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Chuvash",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Haitian",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Ido",
    "Indonesian",
    "Irish",
    "Italian",
    "Japanese",
    "Javanese",
    "Kannada",
    "Kazakh",
    "Kirghiz",
    "Korean",
    "Latin",
    "Latvian",
    "Lithuanian",
    "Lombard",
    "Low Saxon",
    "Luxembourgish",
    "Macedonian",
    "Malagasy",
    "Malay",
    "Malayalam",
    "Marathi",
    "Minangkabau",
    "Nepali",
    "Newar",
    "Norwegian (Bokmal)",
    "Norwegian (Nynorsk)",
    "Occitan",
    "Persian (Farsi)",
    "Piedmontese",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "Scots",
    "Serbian",
    "Serbo-Croatian",
    "Sicilian",
    "Slovak",
    "Slovenian",
    "South Azerbaijani",
    "Spanish",
    "Sundanese",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Tajik",
    "Tamil",
    "Tatar",
    "Telugu",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
    "Volapük",
    "Waray-Waray",
    "Welsh",
    "West Frisian",
    "Western Punjabi",
    "Yoruba",
    "Thai",
    "Mongolian"
]

XLMR_LANGS = [
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian",
    "Assamese", "Azerbaijani", "Basque", "Belarusian", "Bengali",
    "Bengali Romanize", "Bosnian", "Breton", "Bulgarian", "Burmese",
    "Burmese zawgyi font", "Catalan", "Chinese (Simplified)", "Chinese (Traditional)", "Croatian",
    "Czech", "Danish", "Dutch", "English", "Esperanto",
    "Estonian", "Filipino", "Finnish", "French", "Galician",
    "Georgian", "German", "Greek", "Gujarati", "Hausa",
    "Hebrew", "Hindi", "Hindi Romanize", "Hungarian", "Icelandic",
    "Indonesian", "Irish", "Italian", "Japanese", "Javanese",
    "Kannada", "Kazakh", "Khmer", "Korean", "Kurdish (Kurmanji)",
    "Kyrgyz", "Lao", "Latin", "Latvian", "Lithuanian",
    "Macedonian", "Malagasy", "Malay", "Malayalam", "Marathi",
    "Mongolian", "Nepali", "Norwegian", "Oriya", "Oromo",
    "Pashto", "Persian", "Polish", "Portuguese", "Punjabi",
    "Romanian", "Russian", "Sanskrit", "Scottish Gaelic", "Serbian",
    "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali",
    "Spanish", "Sundanese", "Swahili", "Swedish", "Tamil",
    "Tamil Romanize", "Telugu", "Telugu Romanize", "Thai", "Turkish",
    "Ukrainian", "Urdu", "Urdu Romanize", "Uyghur", "Uzbek",
    "Vietnamese", "Welsh", "Western Frisian", "Xhosa", "Yiddish"
]

UN6_LANGS = ["ar", "en", "es", "fr", "ru", "zh"]

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

PHONETIC_SIMILARITY = pd.read_csv(
    "./data/asjp_ldnd_distances_by_iso2.csv", header=0, index_col=0
)


# Dictionary of language and wikisize
language_wikisize = {
    "io": 3, "pms": 3, "scn": 3, "yo": 3,
    "cv": 4, "lmo": 4, "mg": 4, "min": 4, "su": 4, "vo": 4,
    "an": 5, "bar": 5, "br": 5, "ce": 5, "fy": 5, "ga": 5, "gu": 5,
    "is": 5, "jv": 5, "ky": 5, "lb": 5, "mn": 5, "my": 5, "nds": 5,
    "ne": 5, "pa": 5, "pnb": 5, "sw": 5, "tg": 5,
    "af": 6, "ba": 6, "cy": 6, "kn": 6, "la": 6, "mr": 6, "oc": 6,
    "sco": 6, "sq": 6, "tl": 6, "tt": 6, "uz": 6,
    "az": 7, "bn": 7, "bs": 7, "eu": 7, "hi": 7, "ka": 7, "kk": 7,
    "lt": 7, "lv": 7, "mk": 7, "ml": 7, "nn": 7, "ta": 7, "te": 7,
    "ur": 7,
    "ast": 8, "be": 8, "bg": 8, "da": 8, "el": 8, "et": 8, "gl": 8,
    "hr": 8, "hy": 8, "ms": 8, "sh": 8, "sk": 8, "sl": 8, "th": 8,
    "war": 8,
    "fa": 9, "fi": 9, "he": 9, "id": 9, "ko": 9, "no": 9, "ro": 9,
    "sr": 9, "tr": 9, "vi": 9,
    "ar": 10, "ca": 10, "cs": 10, "hu": 10, "nl": 10, "sv": 10,
    "uk": 10,
    "ceb": 11, "it": 11, "ja": 11, "pl": 11, "pt": 11, "zh": 11,
    "de": 12, "es": 12, "fr": 12, "ru": 12,
    "en": 14
}

# Convert to DataFrame
WIKISIZE = pd.DataFrame(list(language_wikisize.items()), columns=["Language", "WikiSize"])

XLMR_SIZE = {
    "af": 1.3,
    "lo": 0.6,
    "am": 0.8,
    "lt": 13.7,
    "ar": 28.0,
    "lv": 8.8,
    "as": 0.1,
    "mg": 0.2,
    "az": 6.5,
    "mk": 4.8,
    "be": 4.3,
    "ml": 7.6,
    "bg": 57.5,
    "mn": 3.0,
    "bn": 8.4,
    "mr": 2.8,
    "ms": 8.5,
    "br": 0.1,
    "my": 1.6,
    "bs": 0.1,
    "ca": 10.1,
    "ne": 3.8,
    "cs": 16.3,
    "nl": 29.3,
    "cy": 0.8,
    "no": 49.0,
    "da": 45.6,
    "om": 0.1,
    "de": 66.6,
    "or": 0.6,
    "el": 46.9,
    "pa": 0.8,
    "en": 300.8,
    "pl": 44.6,
    "eo": 0.9,
    "ps": 0.7,
    "es": 53.3,
    "pt": 49.1,
    "et": 6.1,
    "ro": 61.4,
    "eu": 2.0,
    "ru": 278.0,
    "fa": 111.6,
    "sa": 0.3,
    "fi": 54.3,
    "sd": 0.4,
    "fr": 56.8,
    "si": 3.6,
    "fy": 0.2,
    "sk": 23.2,
    "ga": 0.5,
    "sl": 10.3,
    "gd": 0.1,
    "so": 0.4,
    "gl": 2.9,
    "sq": 5.4,
    "gu": 1.9,
    "sr": 9.1,
    "ha": 0.3,
    "su": 0.1,
    "he": 31.6,
    "sv": 12.1,
    "hi": 20.2,
    "sw": 1.6,
    "ta": 12.2,
    "te": 4.7,
    "hr": 20.5,
    "hu": 58.4,
    "hy": 5.5,
    "id": 148.3,
    "is": 3.2,
    "it": 30.2,
    "ja": 69.3,
    "jv": 0.2,
    "ka": 9.1,
    "kk": 6.4,
    "km": 1.5,
    "kn": 3.3,
    "ko": 54.2,
    "ku": 0.4,
    "ky": 1.2,
    "la": 2.5,
    "th": 71.7,
    "tl": 3.1,
    "tr": 20.9,
    "ug": 0.4,
    "uk": 84.6,
    "ur": 5.7,
    "uz": 0.7,
    "vi": 137.3,
    "xh": 0.1,
    "yi": 0.3,
    "zh": 16.6,
}

# Log2 of XLMR_SIZE for equivalent treatment as BERT_MULTILINGUAL_LANGS
XLMR_SIZE_LOG = {k: math.log2(v) for k, v in XLMR_SIZE.items()}

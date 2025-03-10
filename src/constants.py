from pandas import DataFrame

# languages in XNLI dataset
LANGUAGES = [
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

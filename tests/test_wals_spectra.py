from wals_spectra import get_langs_with_wals_feature, load_wals_feature


def test_langs_with_wals_feature_keyed_by_pipeline_code():
    # 81A: Order of Subject, Object and Verb.
    langs, lang_to_value = get_langs_with_wals_feature("81A")
    wals_values = load_wals_feature("81A")

    assert langs == sorted(lang_to_value)
    # Previously dropped (fr, zh) or matched to the wrong WALS language (de).
    assert lang_to_value["de"] == wals_values["ger"]
    assert lang_to_value["fr"] == wals_values["fre"]
    assert lang_to_value["zh"] == wals_values["mnd"]

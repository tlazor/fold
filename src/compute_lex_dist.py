import csv
import math
import re
import sys
from pycldf import Dataset

# If you have any constants or external mappings:
from constants import FSI_SCALE
import langcodes

###############################################################################
# 1. PARSE WORDS INTO SEGMENTS, HANDLING SPECIAL MARKERS (~, $, ")
###############################################################################
def parse_form(word):
    """
    Convert the raw ASJP-like form into a list of segments, where:
      - A '~' merges the last 2 segments (for two consonants).
      - A '$' merges the last 3 segments (for three consonants).
      - A '"' (double quote) glottalizes the preceding segment.
    """
    tokens = list(word)  # one character per token
    segments = []

    for token in tokens:
        if token == "~":
            # Merge the last 2 segments (e.g. 'k', 'w' => 'kw')
            if len(segments) >= 2:
                s2 = segments.pop()
                s1 = segments.pop()
                segments.append(s1 + s2)
            else:
                # If malformed or insufficient segments, ignore or handle error
                pass

        elif token == "$":
            # Merge the last 3 segments (e.g. 'n', 'd', 'y' => 'ndy')
            if len(segments) >= 3:
                s3 = segments.pop()
                s2 = segments.pop()
                s1 = segments.pop()
                segments.append(s1 + s2 + s3)
            else:
                pass

        elif token == '"':
            # Mark the preceding segment as glottalized.
            # We'll just append ʔ to indicate glottalization
            if segments:
                s = segments.pop()
                segments.append(s + "ʔ")
            else:
                pass
        else:
            # Normal character => push as a new segment
            segments.append(token)

    return segments


###############################################################################
# 2. CUSTOM LEVENSHTEIN ON LISTS OF SEGMENTS
###############################################################################
def custom_levenshtein_distance(w1, w2):
    """
    Standard Levenshtein distance, but operating on lists of segments
    rather than raw strings.
    """
    segs1 = parse_form(w1)
    segs2 = parse_form(w2)

    n, m = len(segs1), len(segs2)
    if n == 0:
        return m
    if m == 0:
        return n

    # dp[i][j] = edit distance between segs1[:i] and segs2[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if segs1[i - 1] == segs2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[n][m]


def ldn(s1, s2):
    """
    Compute a normalized Levenshtein distance using:
        LDN = dist / max_len
    where:
        dist = custom_levenshtein_distance(s1, s2)
        max_len = max(#segments in s1, #segments in s2)
    """
    dist = custom_levenshtein_distance(s1, s2)
    len1 = len(parse_form(s1))
    len2 = len(parse_form(s2))
    max_len = max(len1, len2)
    if dist == 0:
        return 0.0
    return dist / max_len


###############################################################################
# 3. COMPUTE LDND AS RATIO OF SAME-CONCEPT LDN TO DIFF-CONCEPT LDN
###############################################################################
def compute_LDND(langA_concepts, langB_concepts):
    """
    Given two dicts:
       langA_concepts = { concept_id: [forms...], ... }
       langB_concepts = { concept_id: [forms...], ... }

    1) LDN_same = average LDN for form pairs that share the SAME concept.
    2) LDN_diff = average LDN for form pairs that differ in concept.
    3) LDND = LDN_same / LDN_diff

    Returns None if there's not enough data to compute.
    """
    # 1) Same-concept LDN
    same_ldn_values = []
    shared_concepts = set(langA_concepts.keys()).intersection(langB_concepts.keys())
    for c in shared_concepts:
        formsA = langA_concepts[c]
        formsB = langB_concepts[c]
        for fA in formsA:
            for fB in formsB:
                same_ldn_values.append(ldn(fA, fB))

    if not same_ldn_values:
        return None  # No same-concept comparisons
    ldn_same = sum(same_ldn_values) / len(same_ldn_values)

    # 2) Different-concept LDN
    diff_ldn_values = []
    conceptsA = set(langA_concepts.keys())
    conceptsB = set(langB_concepts.keys())
    for cA in conceptsA:
        for cB in conceptsB:
            if cA != cB:
                for fA in langA_concepts[cA]:
                    for fB in langB_concepts[cB]:
                        diff_ldn_values.append(ldn(fA, fB))

    if not diff_ldn_values:
        return None  # No different-concept comparisons
    ldn_diff = sum(diff_ldn_values) / len(diff_ldn_values)
    if ldn_diff == 0:
        return None

    return ldn_same / ldn_diff


###############################################################################
# 4. MAIN SCRIPT: GROUP BY ISO-2 AND COMPUTE LDND
###############################################################################
def main():
    # Path to your local CLDF dataset:
    cldf_metadata_path = "/fold/data/asjp-v20/lexibank-asjp-f0f1d0d/cldf/cldf-metadata.json"
    dataset = Dataset.from_metadata(cldf_metadata_path)

    # Build a list of 3-letter codes from FSI_SCALE plus 'en', etc.
    langs_3letter = [
        langcodes.Language.get(two_letter).to_alpha3()
        for two_letter in list(FSI_SCALE.keys()) + ["en", "hr", "srp"]
    ]

    # The pattern to detect duplicates like "serbian_2", "serbian_3", etc.
    # We'll use it to *ignore* suffixes if you wish. (Kept from your code.)
    pattern = re.compile(r"_(\d+)$")

    # We'll track which language IDs we care about (matching iso3 in langs_3letter).
    # And we'll also store iso2 codes in a map: language_iso2_map[langID] = something
    language_iso2_map = {}
    languages_of_interest = []

    for lang in dataset["LanguageTable"]:
        iso3 = lang.get("ISO639P3code")
        if not iso3:
            continue
        
        # Convert iso3 -> iso2, if possible
        # langcodes.Language.get(iso3).language will typically give
        # the 2-letter code if it exists, else a fallback.
        iso2 = langcodes.Language.get(iso3).language  
        
        if iso3 in langs_3letter:
            # Check if the language's name ends with _\d
            # If it does, we skip it to keep only the "first" name version
            # Or you might keep them all if you prefer.
            lang_name = lang.get("Name", lang["ID"])
            match = pattern.search(lang_name)
            if not match:
                # This is one of the main languages of interest
                languages_of_interest.append(lang["ID"])
                language_iso2_map[lang["ID"]] = iso2

    # Confirm they're valid
    valid_ids = {l["ID"] for l in dataset["LanguageTable"]}
    languages_of_interest = [lid for lid in languages_of_interest if lid in valid_ids]

    # ---------------------------------------------------------------------------
    # Gather forms by (language_id, concept)
    # forms_by_lang[lang_id][concept_id] = [list_of_forms]
    # ---------------------------------------------------------------------------
    forms_by_lang = {}
    for row in dataset["FormTable"]:
        lang_id = row["Language_ID"]
        if lang_id not in languages_of_interest:
            continue
        concept_id = row["Parameter_ID"]
        form = row["Form"]
        forms_by_lang.setdefault(lang_id, {}).setdefault(concept_id, []).append(form)

    # ---------------------------------------------------------------------------
    # Now AGGREGATE all forms by their iso2 code.
    # forms_by_iso2[iso2][concept_id] = [combined list of forms from all lang_ids]
    # ---------------------------------------------------------------------------
    forms_by_iso2 = {}
    for lang_id, concepts_dict in forms_by_lang.items():
        iso2 = language_iso2_map[lang_id]
        for concept_id, form_list in concepts_dict.items():
            forms_by_iso2.setdefault(iso2, {}).setdefault(concept_id, []).extend(form_list)

    # We'll compute distances among these iso2 groups
    iso2_groups = sorted(forms_by_iso2.keys())

    # Build a distance matrix among iso2 codes
    distance_matrix = {code: {} for code in iso2_groups}

    for i, iso2A in enumerate(iso2_groups):
        for iso2B in iso2_groups[i:]:
            ldnd_val = compute_LDND(
                forms_by_iso2.get(iso2A, {}),
                forms_by_iso2.get(iso2B, {})
            )
            distance_matrix[iso2A][iso2B] = ldnd_val
            distance_matrix[iso2B][iso2A] = ldnd_val

    # ---------------------------------------------------------------------------
    # Write CSV: rows/columns are iso2 codes
    # ---------------------------------------------------------------------------
    csv_filename = "data/asjp_ldnd_distances_by_iso2.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header row: empty cell + each iso2
        header = [""] + iso2_groups
        writer.writerow(header)

        for iso2A in iso2_groups:
            row = [iso2A]
            for iso2B in iso2_groups:
                dist = distance_matrix[iso2A][iso2B]
                if dist is None:
                    row.append("")
                else:
                    row.append(f"{dist:.4f}")
            writer.writerow(row)

    print(f"LDND distance matrix (grouped by ISO-2) written to: {csv_filename}")


if __name__ == "__main__":
    main()

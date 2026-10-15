import csv

from pycldf import Dataset

import languages
from paths import DATA_DIR


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
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
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
# 4. MAIN SCRIPT: COMPUTE LDND BETWEEN PIPELINE LANGUAGES
###############################################################################
def main():
    dataset = Dataset.from_metadata(
        DATA_DIR / "asjp-v20" / "lexibank-asjp-f0f1d0d" / "cldf" / "cldf-metadata.json"
    )

    # One pinned ASJP doculect per pipeline language (the standard variety; see
    # data/constants/languages.csv). Matching on ISO codes instead would miss
    # macrolanguages (ASJP tags Mandarin "cmn", not "zho") and pool every
    # dialect that shares a code.
    code_by_doculect = languages.from_source_ids("asjp_id")

    # ---------------------------------------------------------------------------
    # Gather forms by (language code, concept)
    # forms_by_lang[code][concept_id] = [list_of_forms]
    # ---------------------------------------------------------------------------
    forms_by_lang = {}
    for row in dataset["FormTable"]:
        code = code_by_doculect.get(row["Language_ID"])
        if code is None:
            continue
        concept_id = row["Parameter_ID"]
        form = row["Form"]
        forms_by_lang.setdefault(code, {}).setdefault(concept_id, []).append(form)

    codes = sorted(forms_by_lang)

    # Build a distance matrix among language codes
    distance_matrix = {code: {} for code in codes}

    for i, code_a in enumerate(codes):
        for code_b in codes[i:]:
            ldnd_val = compute_LDND(forms_by_lang[code_a], forms_by_lang[code_b])
            distance_matrix[code_a][code_b] = ldnd_val
            distance_matrix[code_b][code_a] = ldnd_val

    # ---------------------------------------------------------------------------
    # Write CSV: rows/columns are pipeline language codes
    # ---------------------------------------------------------------------------
    csv_filename = DATA_DIR / "asjp_ldnd_distances_by_iso2.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header row: empty cell + each code
        writer.writerow([""] + codes)

        for code_a in codes:
            row = [code_a]
            for code_b in codes:
                dist = distance_matrix[code_a][code_b]
                if dist is None:
                    row.append("")
                else:
                    row.append(f"{dist:.4f}")
            writer.writerow(row)

    print(f"LDND distance matrix written to: {csv_filename}")


if __name__ == "__main__":
    main()

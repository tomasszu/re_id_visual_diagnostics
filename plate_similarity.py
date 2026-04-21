import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_CHAR_SCORE = 0.5 # This can be changed, but from some examples this seemed correct

# OCR confusion map
OCR_CONFUSIONS = {
    "0": ["O"], "O": ["0"],
    "1": ["I", "L"], "I": ["1", "L"], "L": ["1", "I"],
    "2": ["Z"], "Z": ["2"],
    "5": ["S"], "S": ["5"],
    "6": ["G"], "G": ["6"],
    "8": ["B"], "B": ["8"],
}


def is_confusion(c1, c2):
    return c1 in OCR_CONFUSIONS and c2 in OCR_CONFUSIONS[c1]


def weighted_edit_distance(
    plate1,
    scores1,
    plate2,
    scores2,
    blank_threshold=MIN_CHAR_SCORE,
    confusion_weight=0.5,
):
    len1, len2 = len(plate1), len(plate2)

    if abs(len1 - len2) > 3:
        return 1

    dp = np.zeros((len1 + 1, len2 + 1))

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            c1, c2 = plate1[i - 1], plate2[j - 1]
            s1 = scores1[i - 1] if i - 1 < len(scores1) else 0.0
            s2 = scores2[j - 1] if j - 1 < len(scores2) else 0.0

            # -------------------------
            # blank handling
            # -------------------------
            if c1 == "_" or c2 == "_":
                conf1 = s1 if c1 == "_" else 1.0
                conf2 = s2 if c2 == "_" else 1.0

                cost = min(1.0, min(conf1, conf2) / blank_threshold)

            # -------------------------
            # normal character match
            # -------------------------
            elif c1 == c2:
                cost = 0.0

            # -------------------------
            # confusion-aware mismatch
            # -------------------------
            else:
                conf = min(s1, s2)

                base_cost = 0.5 + 0.5 * conf # substitution cost starts from min 0.5 for confidence of 0, higher conf -> higher penalty

                if is_confusion(c1, c2):
                    cost = confusion_weight * base_cost
                else:
                    cost = base_cost

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost
            )

    max_len = max(len1, len2)
    return dp[len1][len2] / max_len


def plate_similarity_weighted(track1, track2):
    plate1, scores1 = track1["plate"], np.array(track1["char_scores"])
    plate2, scores2 = track2["plate"], np.array(track2["char_scores"])

    dist = weighted_edit_distance(plate1, scores1, plate2, scores2)
    return max(0.0, 1.0 - dist)




def main():
    # Example track plates
    trackA = {
        "plate": "AB0123",
        "char_scores": [0.99, 0.98, 0.97, 0.95, 0.96, 0.99]
    }

    trackB = {
        "plate": "AB_1_3",  # missing character, simulating OCR uncertainty
        "char_scores": [0.99, 0.98, 0.5, 0.96, 0.42, 0.99]
    }

    trackC = {
        "plate": "A123C",  # shifted character + missing one
        "char_scores": [0.98, 0.95, 0.97, 0.94, 0.80]
    }

    trackD = {
        "plate": "BB_K__3",  # shifted character + missing one
        "char_scores": [0.72, 0.96, 0.45, 0.66, 0.40, 0.49, 0.61]
    }

    trackE = {
        "plate": "KY____6",  # shifted character + missing one
        "char_scores": [0.98, 0.95, 0.45, 0.23, 0.29, 0.34, 0.91]
    }

    trackF = {
        "plate": "ABOIZ3",  # shifted character + missing one
        "char_scores": [0.98, 0.95, 0.92, 0.93, 0.92, 0.98]
    }

    trackX = {
        "plate": "KV534",  # shifted character + missing one
        "char_scores": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    }

    trackY = {
        "plate": "EE8822",  # shifted character + missing one
        "char_scores": [0.99, 0.98, 0.99, 0.99, 0.99, 0.99]
    }

    print("Similarity A vs A:", plate_similarity_weighted(trackA, trackA)) # 1.0
    print("Similarity A vs B:", plate_similarity_weighted(trackA, trackB)) # 0.6933333333333334
    print("Similarity A vs C:", plate_similarity_weighted(trackA, trackC)) # 0.5
    print("Similarity B vs C:", plate_similarity_weighted(trackB, trackC)) # 0.36
    print("Similarity A vs D:", plate_similarity_weighted(trackA, trackD)) # 0.3728571428571429
    print("Similarity B vs D:", plate_similarity_weighted(trackB, trackD)) # 0.3728571428571429
    print("Similarity E vs D:", plate_similarity_weighted(trackE, trackD)) # 0.24857142857142855
    print("Similarity A vs F:", plate_similarity_weighted(trackA, trackF)) # 0.7595833333333334

    print("Similarity X vs Y:", plate_similarity_weighted(trackX, trackY)) # 0.0050000000000000044

    # 0.5 seems like a good cutoff for similarity

if __name__ == "__main__":
    main()
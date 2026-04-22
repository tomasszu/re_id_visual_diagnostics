import numpy as np
from .embedding import load_event_embeddings


# ============================================================
# PLATE HELPERS
# ============================================================
def safe_dominant_plate(plates):
    plates = [p for p in plates if isinstance(p, str) and p.strip()]

    if not plates:
        return None

    return max(set(plates), key=plates.count)


# ============================================================
# GROUP ANALYSIS
# ============================================================
def analyze_group(group, storage):
    plates = []
    times = []
    embeddings = []

    for _, row in group.iterrows():
        lpr = row.get("LPR") or {}

        # high confidence plates only
        if lpr.get("plate") and lpr.get("confidence", 0) > 0.9:
            plates.append(lpr["plate"])

        times.append(row["start_datetime"])
        embeddings.extend(load_event_embeddings(storage, row))

    unique_plates = list(set(plates))
    dominant_plate = safe_dominant_plate(plates)

    span_sec = (
        (max(times) - min(times)).total_seconds()
        if times else 0
    )

    # ---------------- cohesion ----------------
    cohesion = None

    if len(embeddings) >= 2:
        sims = [
            float(np.dot(embeddings[i], embeddings[j]))
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]

        if sims:
            cohesion = float(np.mean(sims))

    return {
        "num_events": len(group),
        "plates": unique_plates,
        "dominant_plate": dominant_plate,
        "plate_conflict": len(unique_plates) > 1,
        "time_span_sec": span_sec,
        "embedding_cohesion": cohesion,
    }


# ============================================================
# BULK ANALYSIS
# ============================================================
def analyze_gt_groups(df, storage):
    results = []

    for gt, group in df.groupby("gt_vehicle_id"):
        res = analyze_group(group, storage)
        res["gt_vehicle_id"] = gt
        results.append(res)

    return results
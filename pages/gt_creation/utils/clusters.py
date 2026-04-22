import numpy as np
from .image import load_image
from .embedding import load_event_embedding


# ============================================================
# BUILD CLUSTERS FROM EVENTS
# ============================================================
def build_clusters(df, storage):
    clusters = {}

    for gt_id, group in df.groupby("gt_vehicle_id"):
        embeddings = []
        plates = []
        times = []
        sample_img = None

        for _, row in group.iterrows():
            vec = load_event_embedding(storage, row)
            if vec is not None:
                embeddings.append(vec)

            lpr = row.get("LPR") or {}
            plate = lpr.get("plate")
            if plate:
                plates.append(plate)

            times.append(row["start_datetime"])

            # first image only
            if sample_img is None:
                rep = row.get("representative") or {}
                sample_img = load_image(storage, rep.get("image_path"))

        if not embeddings or not times:
            continue

        # ---------------- cluster embedding ----------------
        emb_mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(emb_mean)

        if norm > 0:
            emb_mean = emb_mean / norm

        # ---------------- metadata ----------------
        dominant_plate = (
            max(set(plates), key=plates.count)
            if plates else None
        )

        clusters[gt_id] = {
            "gt_id": gt_id,
            "embedding": emb_mean,
            "plate": dominant_plate,
            "num_events": len(group),
            "time_min": min(times),
            "time_max": max(times),
            "img": sample_img,
        }

    return clusters


# ============================================================
# CLUSTER DISTANCE / SIMILARITY
# ============================================================
def cluster_score(a, b):
    emb_sim = float(np.dot(a["embedding"], b["embedding"]))

    plate_score = 1.0 if (
        a["plate"] and b["plate"] and a["plate"] == b["plate"]
    ) else 0.0

    gap = abs((a["time_max"] - b["time_min"]).total_seconds())
    time_score = float(np.exp(-gap / 3600.0))

    score = (
        0.7 * emb_sim +
        0.2 * plate_score +
        0.1 * time_score
    )

    return score, emb_sim, plate_score, time_score


# ============================================================
# MERGE CLUSTERS
# ============================================================
def merge_clusters(storage, df, gt_a, gt_b):
    rows = df[df["gt_vehicle_id"] == gt_b]

    for _, r in rows.iterrows():
        from .ground_truth_utils import assign_event_to_gt
        assign_event_to_gt(storage, r["obj_key"], gt_a)
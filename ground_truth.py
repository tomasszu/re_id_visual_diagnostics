import base64
import io

import numpy as np
import json

from plate_similarity import plate_similarity_weighted


def assign_event_to_gt(storage, event_key, gt_id):
    import json

    data = json.loads(storage.get_object(event_key))

    data.setdefault("ground_truth", {})
    data["ground_truth"]["gt_vehicle_id"] = gt_id
    data["ground_truth"]["assigned"] = True

    storage.put_object(
        event_key,
        json.dumps(data).encode("utf-8")
    )


def create_new_gt_id():
    import uuid
    return f"GT_{uuid.uuid4().hex[:8]}"

def merge_gt_ids(df, source_ids, target_id, storage):
    subset = df[df["gt_vehicle_id"].isin(source_ids)]

    for _, row in subset.iterrows():
        assign_event_to_gt(storage, row["obj_key"], target_id)

def clear_gt_assignment(storage, event_key):
    data = json.loads(storage.get_object(event_key))

    if "ground_truth" in data:
        data["ground_truth"]["assigned"] = False
        data["ground_truth"]["gt_vehicle_id"] = None

    storage.put_object(event_key, json.dumps(data).encode("utf-8"))

# ---------- TIME SCORE ----------
def time_score(e1, e2):
    t1, t2 = e1["start_datetime"], e2["start_datetime"]
    dt = abs((t1 - t2).total_seconds())

    if dt <= 10:
        return 1.0
    if dt >= 300:
        return 0.0

    return 1 - (dt - 10) / (300 - 10)

# ---------- LPR SCORE ----------

def lpr_score(e1, e2):
    l1, l2 = get_lpr(e1), get_lpr(e2)

    if not l1 or not l2:
        return 0.0
    
    p1, p2 = l1.get("plate"), l2.get("plate")
    s1, s2 = l1.get("char_scores"), l2.get("char_scores")

    # ---- HARD VALIDATION ----
    if (
        p1 is None or p2 is None or
        s1 is None or s2 is None or
        len(p1) == 0 or len(p2) == 0
    ):
        return 0.0

    return plate_similarity_weighted(l1, l2)

# ---------- EMBEDDING SCORE ----------
def embedding_score(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))

# ---------- COMBINED ----------
def combined_score(e1, e2, v1, v2):
    ts = time_score(e1, e2)
    ls = lpr_score(e1, e2)
    es = embedding_score(v1, v2)

    return (
        0.3 * ts +
        0.4 * ls +
        0.3 * es,
        ts, ls, es
    )

# ---------- SAFE GET ----------
def get_lpr(e):
    return e.get("LPR") if isinstance(e.get("LPR"), dict) else None

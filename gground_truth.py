import numpy as np
from pyarrow import json


def assign_event_to_gt(storage, event_key, gt_id):
    data = json.loads(storage.get_object(event_key))

    data.setdefault("ground_truth", {})
    data["ground_truth"]["gt_vehicle_id"] = gt_id
    data["ground_truth"]["assigned"] = True

    storage.put_object(event_key, json.dumps(data))

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

    storage.put_object(event_key, json.dumps(data))

def time_score(t1, t2):
    dt = abs((t1 - t2).total_seconds())

    if dt <= 10:
        return 1.0
    if dt >= 300:
        return 0.0

    return 1 - (dt - 10) / (300 - 10)

def lpr_score(lpr1, lpr2):
    if not lpr1 or not lpr2:
        return 0.0

    # placeholder → you plug your logic
    if lpr1["plate"] == lpr2["plate"]:
        return min(lpr1["confidence"], lpr2["confidence"])

    return 0.0

def embedding_score(vec1, vec2):
    return float(np.dot(vec1, vec2))

def combined_score(e1, e2, weights=(0.4, 0.3, 0.3)):
    w_time, w_lpr, w_emb = weights

    return (
        w_time * time_score(e1["start_datetime"], e2["start_datetime"]) +
        w_lpr * lpr_score(e1.get("LPR"), e2.get("LPR")) +
        w_emb * embedding_score(e1["embedding"], e2["embedding"])
    )
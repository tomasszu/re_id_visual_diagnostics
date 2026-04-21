import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import json, io
from datetime import date
from PIL import Image

from minio_backend import MinioBackend
from data_loader import load_embedding
from ground_truth import assign_event_to_gt


# ============================================================
# STORAGE
# ============================================================
def create_storage_from_session():
    cfg = st.session_state["runtime_config"]

    return MinioBackend(
        endpoint=cfg["MINIO_ENDPOINT"],
        access_key=cfg["MINIO_ACCESS_KEY"],
        secret_key=cfg["MINIO_SECRET_KEY"],
        bucket=cfg["MINIO_BUCKET"],
        secure=cfg["MINIO_SECURE"],
    )


# ============================================================
# LOAD
# ============================================================
def discover_days(storage):
    days = set()

    for key in storage.list_objects("enriched_events/"):
        parts = key.split("/")
        if len(parts) < 4:
            continue

        try:
            days.add(date(int(parts[1]), int(parts[2]), int(parts[3])))
        except:
            continue

    return sorted(days)


def load_events(storage, day):
    rows = []

    for key in storage.list_objects(f"enriched_events/{day}"):
        try:
            data = json.loads(storage.get_object(key))
            data["obj_key"] = key
            rows.append(data)
        except:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["start_datetime"] = pd.to_datetime(df["start_timestamp_utc"], errors="coerce")

    df["gt_vehicle_id"] = df.apply(
        lambda x: (x.get("ground_truth") or {}).get("gt_vehicle_id"),
        axis=1
    )

    return df[df["gt_vehicle_id"].notna()]


# ============================================================
# IMAGE
# ============================================================
def load_image(storage, key):
    try:
        if not key:
            return None
        return Image.open(io.BytesIO(storage.get_object(key))).convert("RGB")
    except:
        return None


# ============================================================
# EMBEDDINGS
# ============================================================
def load_event_embedding(storage, row):
    emb = row.get("embedding")
    if not isinstance(emb, dict) or not emb:
        return None

    try:
        return load_embedding(storage, next(iter(emb.values())))
    except:
        return None


# ============================================================
# CLUSTER BUILDING
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
            if lpr.get("plate"):
                plates.append(lpr["plate"])

            times.append(row["start_datetime"])

            if sample_img is None:
                rep = row.get("representative") or {}
                sample_img = load_image(storage, rep.get("image_path"))

        if not embeddings:
            continue

        emb_mean = np.mean(embeddings, axis=0)
        emb_mean = emb_mean / np.linalg.norm(emb_mean)

        dominant_plate = max(set(plates), key=plates.count) if plates else None

        clusters[gt_id] = {
            "gt_id": gt_id,
            "embedding": emb_mean,
            "plate": dominant_plate,
            "num_events": len(group),
            "time_min": min(times),
            "time_max": max(times),
            "img": sample_img
        }

    return clusters


# ============================================================
# SCORING
# ============================================================
def cluster_score(a, b):
    emb_sim = float(np.dot(a["embedding"], b["embedding"]))

    plate_score = 1.0 if a["plate"] and a["plate"] == b["plate"] else 0.0

    # time gap penalty (seconds)
    gap = abs((a["time_max"] - b["time_min"]).total_seconds())
    time_score = np.exp(-gap / 3600.0)  # 1h decay

    return (
        0.7 * emb_sim +
        0.2 * plate_score +
        0.1 * time_score,
        emb_sim,
        plate_score,
        time_score
    )


# ============================================================
# MERGE
# ============================================================
def merge_clusters(storage, df, gt_a, gt_b):
    rows = df[df["gt_vehicle_id"] == gt_b]

    for _, r in rows.iterrows():
        assign_event_to_gt(storage, r["obj_key"], gt_a)


# ============================================================
# UI
# ============================================================
def main():
    st.title("GT Cluster Merge")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO failed")
        return

    # ---------------- LOAD ----------------
    days = discover_days(storage)
    selected_date = st.date_input("Select date", value=max(days))

    df = load_events(storage, selected_date.strftime("%Y/%m/%d"))

    if df.empty:
        st.warning("No GT clusters")
        return

    clusters = build_clusters(df, storage)

    if not clusters:
        st.warning("No valid clusters")
        return

    # ---------------- SELECT BASE ----------------
    st.header("Step 1 — Select Base GT")

    gt_ids = list(clusters.keys())

    selected_gt = st.selectbox("GT cluster", gt_ids)

    base = clusters[selected_gt]

    st.subheader(f"Base GT: {selected_gt}")
    if base["img"]:
        st.image(base["img"], use_container_width=True)

    st.caption(f"Plate: {base['plate']}")
    st.caption(f"Events: {base['num_events']}")

    # ---------------- CANDIDATES ----------------
    st.header("Step 2 — Candidates")

    candidates = []

    for gt_id, c in clusters.items():
        if gt_id == selected_gt:
            continue

        score, emb, plate, time = cluster_score(base, c)

        candidates.append({
            "gt_id": gt_id,
            "score": score,
            "emb": emb,
            "plate": plate,
            "time": time,
            "img": c["img"],
            "events": c["num_events"],
            "plate_val": c["plate"]
        })

    # sorting controls
    sort_by = st.selectbox(
        "Sort by",
        ["score", "emb", "plate", "time"]
    )

    candidates = sorted(candidates, key=lambda x: x[sort_by], reverse=True)

    # ---------------- GRID ----------------
    cols = st.columns(4)

    for i, c in enumerate(candidates[:20]):
        col = cols[i % 4]

        with col:
            if c["img"]:
                st.image(c["img"], use_container_width=True)

            st.caption(f"GT: {c['gt_id']}")
            st.caption(f"Score: {c['score']:.3f}")
            st.caption(f"Emb: {c['emb']:.3f}")
            st.caption(f"Plate: {c['plate_val']}")
            st.caption(f"Events: {c['events']}")

            if st.button("Merge", key=f"merge_{c['gt_id']}"):
                merge_clusters(storage, df, selected_gt, c["gt_id"])
                st.success(f"Merged {c['gt_id']} → {selected_gt}")
                st.rerun()


if __name__ == "__main__":
    main()
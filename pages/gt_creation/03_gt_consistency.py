import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import json, io

from PIL import Image
from datetime import date

from minio_backend import MinioBackend
from data_loader import StorageBackend, load_embedding
from ground_truth import clear_gt_assignment, create_new_gt_id, assign_event_to_gt


# ============================================================
# STORAGE
# ============================================================
def create_storage_from_session() -> StorageBackend:
    cfg = st.session_state["runtime_config"]

    return MinioBackend(
        endpoint=cfg["MINIO_ENDPOINT"],
        access_key=cfg["MINIO_ACCESS_KEY"],
        secret_key=cfg["MINIO_SECRET_KEY"],
        bucket=cfg["MINIO_BUCKET"],
        secure=cfg["MINIO_SECURE"],
    )


# ============================================================
# LOAD EVENTS
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
    prefix = f"enriched_events/{day}"
    rows = []

    for key in storage.list_objects(prefix):
        try:
            raw = storage.get_object(key)
            data = json.loads(raw)
            data["obj_key"] = key
            rows.append(data)
        except:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(rows)
    df["start_datetime"] = pd.to_datetime(df["start_timestamp_utc"], errors="coerce")

    return df

def event_label(row):
    lpr = row.get("LPR") or {}
    plate = lpr.get("plate", "----")

    return (
        f"{row['start_datetime'].strftime('%H:%M:%S')} | "
        f"{plate} | "
        f"{row['vehicle_event_id'][-6:]}"
    )


def extract_gt(row):
    gt = row.get("ground_truth")
    return gt.get("gt_vehicle_id") if isinstance(gt, dict) else None


# ============================================================
# IMAGE
# ============================================================
def load_image(storage, key):
    try:
        img_bytes = storage.get_object(key)
        return Image.open(io.BytesIO(img_bytes))
    except:
        return None


# ============================================================
# EMBEDDINGS
# ============================================================
def load_event_embeddings(storage, event):
    emb = event.get("embedding")
    if not isinstance(emb, dict):
        return []

    out = []
    for v in emb.values():
        try:
            out.append(load_embedding(storage, v))
        except:
            continue

    return [e for e in out if e is not None]


# ============================================================
# ANALYSIS
# ============================================================

def safe_dominant_plate(plates):
    plates = [p for p in plates if isinstance(p, str) and p.strip()]
    if not plates:
        return None
    return max(set(plates), key=plates.count)


def analyze_group(group, storage):
    plates, times, embeddings = [], [], []

    for _, row in group.iterrows():
        lpr = row.get("LPR") or {}

        if lpr.get("plate") and lpr.get("confidence", 0) > 0.9:
            plates.append(lpr["plate"])

        times.append(row["start_datetime"])
        embeddings.extend(load_event_embeddings(storage, row))

    unique_plates = list(set(plates))
    dominant_plate = max(set(plates), key=plates.count) if plates else None

    span_sec = (
        (max(times) - min(times)).total_seconds()
        if times else 0
    )

    cohesion = None
    if len(embeddings) >= 2:
        sims = [
            np.dot(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        cohesion = float(np.mean(sims)) if sims else None

    return {
        "num_events": len(group),
        "plates": unique_plates,
        "dominant_plate": dominant_plate,
        "plate_conflict": len(unique_plates) > 1,
        "time_span_sec": span_sec,
        "embedding_cohesion": cohesion,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("GT Consistency / Conflict Analysis")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO connection failed")
        return

    # ---------------- DAYS ----------------
    days = discover_days(storage)
    if not days:
        st.warning("No data")
        return

    selected_date = st.date_input(
        "Select date",
        value=max(days),
        min_value=min(days),
        max_value=max(days)
    )

    day_str = selected_date.strftime("%Y/%m/%d")

    # ---------------- LOAD ----------------
    df = load_events(storage, day_str)

    if df.empty:
        st.warning("No events")
        return

    df["gt_vehicle_id"] = df.apply(extract_gt, axis=1)
    df = df[df["gt_vehicle_id"].notna()]

    if df.empty:
        st.warning("No GT assigned yet")
        return

    # ---------------- ANALYSIS ----------------
    results = [
        {"gt_vehicle_id": gt, **analyze_group(g, storage)}
        for gt, g in df.groupby("gt_vehicle_id")
    ]

    res_df = pd.DataFrame(results)

    # ---------------- FILTERS ----------------
    st.sidebar.header("Filters")

    only_conflicts = st.sidebar.checkbox("Only plate conflicts", False)
    min_events = st.sidebar.slider("Min events", 1, 20, 2)
    min_emb = st.sidebar.slider("Min embedding cohesion", 0.0, 1.0, 0.5)

    filtered = res_df.copy()

    if only_conflicts:
        filtered = filtered[filtered["plate_conflict"]]

    filtered = filtered[filtered["num_events"] >= min_events]

    filtered = filtered[
        filtered["embedding_cohesion"].isna() |
        (filtered["embedding_cohesion"] >= min_emb)
    ]

    # ---------------- TABLE ----------------
    st.header("GT Clusters")
    st.dataframe(filtered, use_container_width=True)

    if filtered.empty:
        return

    # ---------------- DRILLDOWN ----------------
    selected_gt = st.selectbox("Select GT ID", filtered["gt_vehicle_id"])

    group = df[df["gt_vehicle_id"] == selected_gt].sort_values("start_datetime")

    st.subheader(f"GT: {selected_gt}")
    st.write("Events:", len(group))

    # ============================================================
    # SELECTION
    # ============================================================
    selected_indices = st.multiselect(
        "Select events",
        options=list(group.index),
        format_func=lambda i: event_label(group.loc[i])
    )

    # ============================================================
    # ACTIONS
    # ============================================================
    st.subheader("GT Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Remove selected events from GT"):
            for i in selected_indices:
                row = group.loc[i]
                clear_gt_assignment(storage, row["obj_key"])

            st.success("Events unassigned")
            st.rerun()

    with col2:
        if st.button("Split into new GT"):
            if not selected_indices:
                st.warning("No events selected")
            else:
                new_gt = create_new_gt_id()

                for i in selected_indices:
                    row = group.loc[i]
                    assign_event_to_gt(storage, row["obj_key"], new_gt)

                st.success(f"Created new GT: {new_gt}")
                st.rerun()

    # ============================================================
    # DRILLDOWN VISUALS
    # ============================================================
    cols = st.columns(6)

    for i, (_, row) in enumerate(group.iterrows()):
        col = cols[i % 6]

        rep = row.get("representative")
        if not isinstance(rep, dict):
            continue

        img = load_image(storage, rep.get("image_path"))
        if img is None:
            continue

        col.image(img, use_container_width=True, caption=None)
        lpr = row.get("LPR") or {}

        col.caption(
            f"{row['start_datetime'].strftime('%H:%M:%S')} | "
            f"{lpr.get('plate', '----')}"
        )


# ============================================================
if __name__ == "__main__":
    main()
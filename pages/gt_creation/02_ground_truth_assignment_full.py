# ============================================================
# CHANGELOG
# ============================================================
# 2026-04-21 FIX REVISION
# - Fixed preview rendering crash (None-safe base64 fallback)
# - Removed index-based candidate lookup (replaced with event_id mapping)
# - Stabilized visual comparison pipeline (no dataframe positional coupling)
# - Hardened image loading (missing/invalid path safe)
# - Ensured suggestions table remains scalar-only and deterministic
# ============================================================

import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import json, io, base64
from datetime import date
from PIL import Image

from data_loader import StorageBackend, load_embedding
from minio_backend import MinioBackend
from ground_truth import (
    assign_event_to_gt,
    create_new_gt_id,
    combined_score
)

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
# IMAGE SAFE PIPELINE
# ============================================================
def load_image(storage, key):
    try:
        if not key:
            return None
        img_bytes = storage.get_object(key)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        return None


def image_to_b64(img, size=(120, 120)):
    if img is None:
        return ""

    try:
        img = img.copy()
        img.thumbnail(size)

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except:
        return ""


# ============================================================
# DATA
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

    df["is_assigned"] = df["gt_vehicle_id"].notna()

    return df


# ============================================================
# EMBEDDING
# ============================================================
def load_event_embedding(storage, event):
    emb = event.get("embedding")
    if not isinstance(emb, dict) or not emb:
        return None

    try:
        emb_info = next(iter(emb.values()))
        return load_embedding(storage, emb_info)
    except:
        return None


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("Ground Truth Assignment")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO connection failed")
        return

    # ---------------- DATA ----------------
    days = discover_days(storage)
    selected_date = st.date_input("Select date", value=max(days))

    df = load_events(storage, selected_date.strftime("%Y/%m/%d"))

    if df.empty:
        st.warning("No events")
        return

    # ============================================================
    # UNASSIGNED TABLE
    # ============================================================
    st.header("Unassigned Events")

    unassigned = df[~df["is_assigned"]]

    if unassigned.empty:
        st.success("All assigned")
        return

    rows = []

    for _, row in unassigned.iterrows():
        lpr = row.get("LPR") or {}

        img = load_image(storage, row["representative"]["image_path"])

        rows.append({
            "preview": image_to_b64(img),
            "time": row["start_datetime"].strftime("%H:%M:%S"),
            "camera": row["camera_id"],
            "plate": lpr.get("plate"),
            "conf": lpr.get("confidence"),
            "event_id": row["vehicle_event_id"]
        })

    table_df = pd.DataFrame(rows)

    table = st.dataframe(
        table_df,
        column_config={
            "preview": st.column_config.ImageColumn("preview")
        },
        selection_mode="single-row",
        on_select="rerun",
        use_container_width=True
    )

    if not table.selection.rows:
        return

    selected_event_id = table_df.iloc[table.selection.rows[0]]["event_id"]
    query = df[df["vehicle_event_id"] == selected_event_id].iloc[0]

    query_vec = load_event_embedding(storage, query)

    if query_vec is None:
        st.error("Missing embedding")
        return

# ============================================================
# CANDIDATES
# ============================================================
    st.header("Suggestions")

    candidates = []

    for _, row in df.iterrows():
        if row["vehicle_event_id"] == selected_event_id:
            continue

        vec2 = load_event_embedding(storage, row)
        if vec2 is None:
            continue

        score, ts, ls, es = combined_score(query, row, query_vec, vec2)

        img = load_image(storage, row["representative"]["image_path"])

        candidates.append({
            "preview": image_to_b64(img),
            "event_id": row["vehicle_event_id"],
            "score": round(score, 3),
            "ts": round(ts, 3),
            "ls": round(ls, 3),
            "es": round(es, 3),
            "gt": row.get("gt_vehicle_id"),
            "plate": (row.get("LPR") or {}).get("plate")
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:10]
    cand_df = pd.DataFrame(candidates)

    st.dataframe(
        cand_df,
        column_config={
            "preview": st.column_config.ImageColumn("preview")  # ✅ SAME AS UNASSIGNED
        },
        use_container_width=True
    )

    # ============================================================
    # STABLE COMPARISON (NO INDEX COUPLING)
    # ============================================================
    st.subheader("Visual Comparison")

    selected_cand_id = st.selectbox(
        "Candidate event",
        cand_df["event_id"].tolist(),
        key="selected_cand_id"
    )

    cand_event = df[df["vehicle_event_id"] == selected_cand_id].iloc[0]

    q_img = load_image(storage, query["representative"]["image_path"])
    c_img = load_image(storage, cand_event["representative"]["image_path"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("QUERY")
        if q_img:
            st.image(q_img, use_container_width=True)

    with col2:
        st.write("CANDIDATE")
        if c_img:
            st.image(c_img, use_container_width=True)

    # ============================================================
    # ASSIGNMENT
    # ============================================================
    st.header("Assign GT")

    gt_options = sorted(df["gt_vehicle_id"].dropna().unique())

    # get GT of currently selected candidate
    selected_cand_row = df[df["vehicle_event_id"] == selected_cand_id]
    selected_cand_gt = (
        selected_cand_row.iloc[0].get("gt_vehicle_id")
        if not selected_cand_row.empty else None
    )

    # build display labels
    gt_display = []
    for gt in gt_options:
        if selected_cand_gt and gt == selected_cand_gt:
            gt_display.append(f"{gt}  ←  [ current chosen candidate]")
        else:
            gt_display.append(gt)

    col1, col2 = st.columns(2)

    with col1:
        gt_choice = st.selectbox("GT", gt_display)

        # strip annotation safely
        selected_gt = gt_choice.split("  ←")[0]

        if st.button("Assign"):
            assign_event_to_gt(storage, query["obj_key"], selected_gt)
            st.rerun()

    with col2:
        if st.button("Assign NEW"):
            new_id = create_new_gt_id()
            assign_event_to_gt(storage, query["obj_key"], new_id)
            st.rerun()


if __name__ == "__main__":
    main()
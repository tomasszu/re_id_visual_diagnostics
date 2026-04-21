import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import io, json
from datetime import date
from PIL import Image

from data_loader import load_embedding
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
# IMAGE
# ============================================================
def load_image(storage, key):
    try:
        if not key:
            return None
        img_bytes = storage.get_object(key)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        return None

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
# STATE
# ============================================================
def init_state():
    if "selected_event_id" not in st.session_state:
        st.session_state.selected_event_id = None

    if "selected_candidate_id" not in st.session_state:
        st.session_state.selected_candidate_id = None


# ============================================================
# GT RESOLVE
# ============================================================
def resolve_gt(row):
    return (row.get("ground_truth") or {}).get("gt_vehicle_id")


# ============================================================
# GT UNION MERGE (CORE FIX)
# ============================================================
def merge_gt(storage, df, query_row, cand_row):
    query_gt = resolve_gt(query_row)
    cand_gt = resolve_gt(cand_row)

    # --------------------------------------------------------
    # CASE 1: both unassigned → create new GT
    # --------------------------------------------------------
    if not query_gt and not cand_gt:
        new_gt = create_new_gt_id()
        assign_event_to_gt(storage, query_row["obj_key"], new_gt)
        assign_event_to_gt(storage, cand_row["obj_key"], new_gt)
        return

    # --------------------------------------------------------
    # CASE 2: query has GT, candidate doesn't
    # --------------------------------------------------------
    if query_gt and not cand_gt:
        assign_event_to_gt(storage, cand_row["obj_key"], query_gt)
        return

    # --------------------------------------------------------
    # CASE 3: candidate has GT, query doesn't
    # --------------------------------------------------------
    if cand_gt and not query_gt:
        assign_event_to_gt(storage, query_row["obj_key"], cand_gt)
        return

    # --------------------------------------------------------
    # CASE 4: both have GT → MERGE CLUSTERS
    # --------------------------------------------------------
    if query_gt and cand_gt and query_gt != cand_gt:

        for _, r in df.iterrows():
            r_gt = resolve_gt(r)
            if r_gt == cand_gt:
                assign_event_to_gt(storage, r["obj_key"], query_gt)


# ============================================================
# UI COMPONENTS
# ============================================================
def card(event_id, img, labels, key_prefix, state_key):
    with st.container(border=True):
        if img:
            st.image(img, use_container_width=True)

        for l in labels:
            st.caption(l)

        if st.button("Select", key=f"{key_prefix}_{event_id}"):
            st.session_state[state_key] = event_id
            st.rerun()


def grid(items, cols, key_prefix, state_key):
    chunks = [items[i:i + cols] for i in range(0, len(items), cols)]

    for chunk in chunks:
        cols_ui = st.columns(cols)

        for i, item in enumerate(chunk):
            with cols_ui[i]:
                card(
                    item["event_id"],
                    item["img"],
                    item["labels"],
                    key_prefix,
                    state_key
                )


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("GT Labeling – 2 Click Clustering Mode")

    init_state()

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO connection failed")
        return

    # ============================================================
    # LOAD DATA
    # ============================================================
    days = discover_days(storage)
    selected_date = st.date_input("Select date", value=max(days))

    df = load_events(storage, selected_date.strftime("%Y/%m/%d"))

    if df.empty:
        st.warning("No data")
        return

    unassigned = df[~df["is_assigned"]]

    if unassigned.empty:
        st.success("All assigned")
        return

    # ============================================================
    # STEP 1: SELECT QUERY
    # ============================================================
    st.header("Step 1 — Select Event")

    event_cards = []
    for _, row in unassigned.iterrows():
        lpr = row.get("LPR") or {}

        img = load_image(storage, row["representative"]["image_path"])

        event_cards.append({
            "event_id": row["vehicle_event_id"],
            "img": img,
            "labels": [
                row["start_datetime"].strftime("%H:%M:%S"),
                f"Cam: {row['camera_id']}",
                f"Plate: {lpr.get('plate')}",
            ]
        })

    grid(event_cards, cols=3, key_prefix="event", state_key="selected_event_id")

    if not st.session_state.selected_event_id:
        st.stop()

    query = df[df["vehicle_event_id"] == st.session_state.selected_event_id].iloc[0]
    query_vec = load_event_embedding(storage, query)

    q_img = load_image(storage, query["representative"]["image_path"])

    st.subheader("Query Event")
    st.image(q_img, use_container_width=True)

    # ============================================================
    # STEP 2: CANDIDATES
    # ============================================================
    st.header("Step 2 — Pick Match (auto merges clusters)")

    candidates = []

    for _, row in df.iterrows():
        if row["vehicle_event_id"] == st.session_state.selected_event_id:
            continue

        vec2 = load_event_embedding(storage, row)
        if vec2 is None:
            continue

        score, ts, ls, es = combined_score(query, row, query_vec, vec2)

        img = load_image(storage, row["representative"]["image_path"])

        candidates.append({
            "event_id": row["vehicle_event_id"],
            "score": score,
            "img": img,
            "labels": [
                f"Overall Score:{score:.3f}",
                f"Embedding Score:{es:.3f}",
                f"Plate:{(row.get('LPR') or {}).get('plate')}"
            ]
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:12]

    grid(candidates, cols=4, key_prefix="cand", state_key="selected_candidate_id")

    # ============================================================
    # STEP 3: APPLY MERGE
    # ============================================================
    if st.session_state.selected_candidate_id:

        cand = df[df["vehicle_event_id"] == st.session_state.selected_candidate_id].iloc[0]

        merge_gt(storage, df, query, cand)

        # reset workflow
        st.session_state.selected_event_id = None
        st.session_state.selected_candidate_id = None

        st.rerun()


if __name__ == "__main__":
    main()
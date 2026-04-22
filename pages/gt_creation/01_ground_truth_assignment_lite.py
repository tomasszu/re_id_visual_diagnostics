import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd

from pages.gt_creation.utils import (
    create_storage_from_session,
    load_events,
    discover_days,
    load_image,
    load_event_embedding,
    merge_gt
)

from ground_truth import combined_score


# ============================================================
# STATE
# ============================================================
def init_state():
    if "selected_event_id" not in st.session_state:
        st.session_state.selected_event_id = None

    if "selected_candidate_id" not in st.session_state:
        st.session_state.selected_candidate_id = None


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

    if not days:
        st.warning("No data")
        return

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
        rep = row.get("representative") or {}

        img = load_image(storage, rep.get("image_path"))

        event_cards.append({
            "event_id": row["vehicle_event_id"],
            "img": img,
            "labels": [
                row["start_datetime"].strftime("%H:%M:%S"),
                f"Cam: {row.get('camera_id')}",
                f"Plate: {lpr.get('plate')}",
            ]
        })

    grid(event_cards, cols=3, key_prefix="event", state_key="selected_event_id")

    if not st.session_state.selected_event_id:
        st.stop()

    query = df[df["vehicle_event_id"] == st.session_state.selected_event_id].iloc[0]
    query_vec = load_event_embedding(storage, query)

    q_img = load_image(storage, query["representative"].get("image_path"))

    st.subheader("Query Event")

    if q_img:
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

        rep = row.get("representative") or {}
        lpr = row.get("LPR") or {}

        img = load_image(storage, rep.get("image_path"))

        candidates.append({
            "event_id": row["vehicle_event_id"],
            "score": score,
            "img": img,
            "labels": [
                f"Overall Score:{score:.3f}",
                f"Embedding Score:{es:.3f}",
                f"Plate:{lpr.get('plate')}"
            ]
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:12]

    grid(candidates, cols=4, key_prefix="cand", state_key="selected_candidate_id")

    # ============================================================
    # STEP 3: APPLY MERGE
    # ============================================================
    if st.session_state.selected_candidate_id:

        cand = df[
            df["vehicle_event_id"] == st.session_state.selected_candidate_id
        ].iloc[0]

        merge_gt(storage, df, query, cand)

        st.session_state.selected_event_id = None
        st.session_state.selected_candidate_id = None

        st.rerun()


if __name__ == "__main__":
    main()
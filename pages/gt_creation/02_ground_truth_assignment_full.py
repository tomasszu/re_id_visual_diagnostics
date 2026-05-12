# ============================================================
# CHANGELOG
# ============================================================
# 2026-04-21 FIX REVISION
# - Fixed preview rendering crash (None-safe base64 fallback)
# - Removed index-based candidate lookup (replaced with event_id mapping)
# - Stabilized visual comparison pipeline (no dataframe positional coupling)
# - Hardened image loading (missing/invalid path safe)
# - Ensured suggestions table remains scalar-only and deterministic
#
# 2026-05-12 MODULARIZATION REVISION
# - Removed duplicated storage initialization logic
# - Removed duplicated event loading / normalization pipeline
# - Removed duplicated image loading and preview encoding utilities
# - Removed duplicated embedding loading logic
# - Standardized event schema through utils.load_events()
# - Centralized GT-related operations into utils package
# - Page converted into UI/controller layer only
# - Reduced risk of schema drift between annotator pages
# - Future event schema changes now isolated to utils/data.py
# ============================================================

import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd

from pages.gt_creation.utils import (
    create_storage_from_session,
    load_events,
    discover_days,
    load_image,
    image_to_b64,
    load_event_embedding,
)

from ground_truth import (
    assign_event_to_gt,
    create_new_gt_id,
    combined_score
)


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("Ground Truth Assignment")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO connection failed")
        return

    # ============================================================
    # DATA
    # ============================================================
    days = discover_days(storage)

    if not days:
        st.warning("No events")
        return

    selected_date = st.date_input(
        "Select date",
        value=max(days)
    )

    df = load_events(
        storage,
        selected_date.strftime("%Y/%m/%d")
    )

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
        rep = row.get("representative") or {}

        img = load_image(storage, rep.get("image_path"))

        rows.append({
            "preview": image_to_b64(img),
            "time": row["start_datetime"].strftime("%H:%M:%S"),
            "camera": row.get("camera_id"),
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

    selected_event_id = (
        table_df.iloc[table.selection.rows[0]]["event_id"]
    )

    query = df[
        df["vehicle_event_id"] == selected_event_id
    ].iloc[0]

    query_vec = load_event_embedding(storage, query)

    if query_vec is None:
        st.error("Missing embedding")
        return

    # ============================================================
    # BULK INITIALIZATION
    # ============================================================
    st.subheader("Bulk Initialization")

    if st.button("Bulk Initialize GT for vehicles with >1 events"):

        grouped = unassigned.groupby("vehicle_id")

        created = 0

        for _, group in grouped:

            if len(group) < 2:
                continue

            new_gt = create_new_gt_id()

            for _, row in group.iterrows():
                assign_event_to_gt(
                    storage,
                    row["obj_key"],
                    new_gt
                )

            created += 1

        st.success(f"Created {created} GT IDs")

        st.cache_data.clear()
        st.rerun()

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

        score, ts, ls, es = combined_score(
            query,
            row,
            query_vec,
            vec2
        )

        rep = row.get("representative") or {}
        lpr = row.get("LPR") or {}

        img = load_image(storage, rep.get("image_path"))

        candidates.append({
            "preview": image_to_b64(img),
            "event_id": row["vehicle_event_id"],
            "score": round(score, 3),
            "ts": round(ts, 3),
            "ls": round(ls, 3),
            "es": round(es, 3),
            "gt": row.get("gt_vehicle_id"),
            "plate": lpr.get("plate")
        })

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )[:10]

    cand_df = pd.DataFrame(candidates)

    st.dataframe(
        cand_df,
        column_config={
            "preview": st.column_config.ImageColumn("preview")
        },
        use_container_width=True
    )

    if cand_df.empty:
        st.warning("No candidates")
        return

    # ============================================================
    # VISUAL COMPARISON
    # ============================================================
    st.subheader("Visual Comparison")

    selected_cand_id = st.selectbox(
        "Candidate event",
        cand_df["event_id"].tolist(),
        key="selected_cand_id"
    )

    cand_event = df[
        df["vehicle_event_id"] == selected_cand_id
    ].iloc[0]

    q_rep = query.get("representative") or {}
    c_rep = cand_event.get("representative") or {}

    q_img = load_image(storage, q_rep.get("image_path"))
    c_img = load_image(storage, c_rep.get("image_path"))

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

    gt_options = sorted(
        df["gt_vehicle_id"]
        .dropna()
        .unique()
    )

    selected_cand_gt = cand_event.get("gt_vehicle_id")

    gt_display = []

    for gt in gt_options:

        if selected_cand_gt and gt == selected_cand_gt:
            gt_display.append(
                f"{gt}  ←  [ current chosen candidate]"
            )
        else:
            gt_display.append(gt)

    col1, col2 = st.columns(2)

    with col1:

        if gt_display:

            gt_choice = st.selectbox(
                "GT",
                gt_display
            )

            selected_gt = gt_choice.split("  ←")[0]

            if st.button("Assign"):

                assign_event_to_gt(
                    storage,
                    query["obj_key"],
                    selected_gt
                )

                st.cache_data.clear()
                st.rerun()

        else:
            st.info("No existing GT IDs yet")

    with col2:

        if st.button("Assign NEW"):

            new_id = create_new_gt_id()

            assign_event_to_gt(
                storage,
                query["obj_key"],
                new_id
            )

            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
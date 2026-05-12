# ============================================================
# CHANGELOG
# ============================================================
# 2026-05-12 INITIAL MODULARIZATION REVISION
#
# ARCHITECTURE
# - Converted page into UI/controller layer only
# - Removed duplicated storage initialization logic
# - Removed duplicated event loading / normalization pipeline
# - Removed duplicated image loading logic
# - Removed duplicated embedding loading pipeline
# - Removed duplicated GT extraction logic
# - Centralized event schema through utils.load_events()
#
# ANALYSIS
# - Centralized GT consistency analysis into utils.analysis
# - Centralized dominant plate resolution logic
# - Centralized multi-embedding loading pipeline
# - Standardized cohesion calculation behavior across pages
#
# GT OPERATIONS
# - Centralized GT mutation operations into utils.ground_truth_utils
# - Added reusable split_events_to_new_gt() operation
# - Added reusable remove_events_from_gt() operation
#
# MAINTAINABILITY
# - Reduced risk of schema drift between pages
# - Future schema changes isolated to utils/data.py
# - Future embedding changes isolated to utils/embedding.py
# - Future GT logic changes isolated to utils/ground_truth_utils.py
# ============================================================

import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd

from pages.gt_creation.utils import (
    create_storage_from_session,
    discover_days,
    load_events,
    event_label,
    load_image,
    analyze_gt_groups,
    remove_events_from_gt,
    split_events_to_new_gt,
)


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("GT Consistency / Conflict Analysis")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO connection failed")
        return

    # ============================================================
    # DAYS
    # ============================================================
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

    # ============================================================
    # LOAD
    # ============================================================
    df = load_events(
        storage,
        selected_date.strftime("%Y/%m/%d")
    )

    if df.empty:
        st.warning("No events")
        return

    # standardized GT schema already exists from utils.load_events()
    df = df[df["gt_vehicle_id"].notna()]

    if df.empty:
        st.warning("No GT assigned yet")
        return

    # ============================================================
    # ANALYSIS
    # ============================================================
    results = analyze_gt_groups(df, storage)

    res_df = pd.DataFrame(results)

    # ============================================================
    # FILTERS
    # ============================================================
    st.sidebar.header("Filters")

    only_conflicts = st.sidebar.checkbox(
        "Only plate conflicts",
        False
    )

    min_events = st.sidebar.slider(
        "Min events",
        1,
        20,
        2
    )

    min_emb = st.sidebar.slider(
        "Min embedding cohesion",
        0.0,
        1.0,
        0.5
    )

    filtered = res_df.copy()

    if only_conflicts:
        filtered = filtered[
            filtered["plate_conflict"]
        ]

    filtered = filtered[
        filtered["num_events"] >= min_events
    ]

    filtered = filtered[
        filtered["embedding_cohesion"].isna() |
        (filtered["embedding_cohesion"] >= min_emb)
    ]

    # ============================================================
    # TABLE
    # ============================================================
    st.header("GT Clusters")

    st.dataframe(
        filtered,
        use_container_width=True
    )

    if filtered.empty:
        return

    # ============================================================
    # DRILLDOWN
    # ============================================================
    selected_gt = st.selectbox(
        "Select GT ID",
        filtered["gt_vehicle_id"]
    )

    group = df[
        df["gt_vehicle_id"] == selected_gt
    ].sort_values("start_datetime")

    st.subheader(f"GT: {selected_gt}")

    st.write(
        "Events:",
        len(group)
    )

    # ============================================================
    # EVENT SELECTION
    # ============================================================
    selected_indices = st.multiselect(
        "Select events",
        options=list(group.index),
        format_func=lambda i: event_label(group.loc[i])
    )

    selected_rows = group.loc[selected_indices]

    # ============================================================
    # GT ACTIONS
    # ============================================================
    st.subheader("GT Actions")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Remove selected events from GT"):

            remove_events_from_gt(
                storage,
                selected_rows
            )

            st.success("Events unassigned")
            st.rerun()

    with col2:

        if st.button("Split into new GT"):

            if selected_rows.empty:
                st.warning("No events selected")

            else:
                new_gt = split_events_to_new_gt(
                    storage,
                    selected_rows
                )

                st.success(
                    f"Created new GT: {new_gt}"
                )

                st.rerun()

    # ============================================================
    # DRILLDOWN VISUALS
    # ============================================================
    cols = st.columns(6)

    for i, (_, row) in enumerate(group.iterrows()):

        col = cols[i % 6]

        rep = row.get("representative") or {}

        img = load_image(
            storage,
            rep.get("image_path")
        )

        if img is None:
            continue

        col.image(
            img,
            use_container_width=True,
            caption=None
        )

        lpr = row.get("LPR") or {}

        col.caption(
            f"{row['start_datetime'].strftime('%H:%M:%S')} | "
            f"{lpr.get('plate', '----')}"
        )


# ============================================================
if __name__ == "__main__":
    main()
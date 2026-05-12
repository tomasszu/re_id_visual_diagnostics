# ============================================================
# CHANGELOG
# ============================================================
# 2026-05-12 INITIAL MODULARIZATION REVISION
#
# PURPOSE
# - Refactored GT cluster merge page to use shared utils package.
# - Removed duplicated storage/image/data/embedding/cluster logic.
# - Standardized cluster schema and loading pipeline across pages.
#
# ARCHITECTURE
# - Shared logic moved into:
#     utils.storage
#     utils.data
#     utils.image
#     utils.clusters
#
# PAGE RESPONSIBILITY AFTER REFACTOR
# - This page now focuses only on:
#     - UI rendering
#     - user interaction
#     - candidate sorting
#     - merge triggering
#
# CENTRALIZED PIPELINES
# - Event normalization handled by utils.data.load_events()
# - GT extraction handled centrally
# - Image loading handled centrally
# - Cluster construction handled centrally
# - Cluster scoring handled centrally
# - Cluster merge execution handled centrally
#
# RESULT
# - Reduced maintenance burden
# - Eliminated repeated cluster-analysis implementations
# - Prevents schema drift between pages
# - Safer future modifications to GT clustering logic
# ============================================================

import streamlit as st
st.set_page_config(layout="wide")

from pages.gt_creation.utils import (
    create_storage_from_session,
    discover_days,
    load_events,
    build_clusters,
    cluster_score,
    merge_clusters,
)


# ============================================================
# MAIN
# ============================================================
def main():
    st.title("GT Cluster Merge")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("MinIO failed")
        return

    # ============================================================
    # LOAD
    # ============================================================
    days = discover_days(storage)

    if not days:
        st.warning("No data")
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
        st.warning("No GT clusters")
        return

    df = df[df["gt_vehicle_id"].notna()]

    if df.empty:
        st.warning("No GT clusters")
        return

    # ============================================================
    # BUILD CLUSTERS
    # ============================================================
    clusters = build_clusters(df, storage)

    if not clusters:
        st.warning("No valid clusters")
        return

    # ============================================================
    # SELECT BASE CLUSTER
    # ============================================================
    st.header("Step 1 — Select Base GT")

    gt_ids = sorted(clusters.keys())

    selected_gt = st.selectbox(
        "GT cluster",
        gt_ids
    )

    base = clusters[selected_gt]

    st.subheader(f"Base GT: {selected_gt}")

    if base["img"]:
        st.image(base["img"], use_container_width=True)

    st.caption(f"Plate: {base['plate']}")
    st.caption(f"Events: {base['num_events']}")

    # ============================================================
    # CANDIDATES
    # ============================================================
    st.header("Step 2 — Candidates")

    candidates = []

    for gt_id, cluster in clusters.items():

        if gt_id == selected_gt:
            continue

        score, emb, plate, time_score = cluster_score(
            base,
            cluster
        )

        candidates.append({
            "gt_id": gt_id,
            "score": score,
            "emb": emb,
            "plate": plate,
            "time": time_score,
            "img": cluster["img"],
            "events": cluster["num_events"],
            "plate_val": cluster["plate"]
        })

    sort_by = st.selectbox(
        "Sort by",
        ["score", "emb", "plate", "time"]
    )

    candidates = sorted(
        candidates,
        key=lambda x: x[sort_by],
        reverse=True
    )

    # ============================================================
    # CANDIDATE GRID
    # ============================================================
    cols = st.columns(4)

    for i, candidate in enumerate(candidates[:20]):

        col = cols[i % 4]

        with col:

            if candidate["img"]:
                st.image(
                    candidate["img"],
                    use_container_width=True
                )

            st.caption(f"GT: {candidate['gt_id']}")
            st.caption(f"Score: {candidate['score']:.3f}")
            st.caption(f"Emb: {candidate['emb']:.3f}")
            st.caption(f"Plate: {candidate['plate_val']}")
            st.caption(f"Events: {candidate['events']}")

            if st.button(
                "Merge",
                key=f"merge_{candidate['gt_id']}"
            ):

                merge_clusters(
                    storage,
                    df,
                    selected_gt,
                    candidate["gt_id"]
                )

                st.success(
                    f"Merged {candidate['gt_id']} → {selected_gt}"
                )

                st.rerun()


# ============================================================
if __name__ == "__main__":
    main()
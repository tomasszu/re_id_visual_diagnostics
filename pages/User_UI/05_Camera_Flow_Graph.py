import streamlit as st
import pandas as pd
import os, json, tempfile
import math
import colorsys
from data_loader import StorageBackend
from minio_backend import MinioBackend
from pyvis.network import Network
import streamlit.components.v1 as components



# ---------------- STORAGE ----------------
def create_storage_from_session() -> StorageBackend:
    cfg = st.session_state["runtime_config"]

    return MinioBackend(
        endpoint=cfg["MINIO_ENDPOINT"],
        access_key=cfg["MINIO_ACCESS_KEY"],
        secret_key=cfg["MINIO_SECRET_KEY"],
        bucket=cfg["MINIO_BUCKET"],
        secure=cfg["MINIO_SECURE"],
    )


# ---------------- LOAD ----------------
@st.cache_data
def load_events(_storage, day):
    rows = []

    for key in _storage.list_objects(f"vehicle_events/{day}"):
        raw = _storage.get_object(key)
        rows.append(json.loads(raw))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["end_timestamp_utc"], errors="coerce")
    return df.sort_values("datetime")


# ---------------- TRANSITIONS ----------------
def build_transitions(df, min_score=0.0, max_gap=None):
    transitions = {}

    for vid, group in df.groupby("vehicle_id"):
        group = group.sort_values("datetime")
        prev_row = None

        for row in group.itertuples():
            if row.reid_score < min_score:
                continue

            if prev_row is not None:
                gap = (row.datetime - prev_row.datetime).total_seconds()
                if max_gap and gap > max_gap:
                    prev_row = row
                    continue

                if prev_row.camera_id != row.camera_id:
                    key = (prev_row.camera_id, row.camera_id)
                    transitions[key] = transitions.get(key, 0) + 1

            prev_row = row

    return transitions


def transitions_to_df(transitions):
    return pd.DataFrame([
        {"from": k[0], "to": k[1], "count": v}
        for k, v in transitions.items()
    ]).sort_values("count", ascending=False)


# ---------------- GRAPH ----------------
def assign_edge_colors(transitions):
    """
    Assign distinct bright colors for dark background.
    """
    edges = list(transitions.keys())
    n = len(edges)
    edge_colors = {}

    for i, edge in enumerate(edges):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)  # brighter saturation and value
        color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        edge_colors[edge] = color

    return edge_colors


def render_graph(transitions):
    net = Network(height="650px", width="100%", directed=True, bgcolor="#111111", font_color="white")

    # collect cameras
    cameras = set()
    for (src, dst) in transitions.keys():
        cameras.add(src)
        cameras.add(dst)

    # add nodes
    for cam in cameras:
        net.add_node(
            cam,
            label=cam,
            shape="dot",
            size=20,
            color="#1f78b4",  # blueish for night mode
            font={"color": "white"}
        )

    # --- GLOBAL NORMALIZATION ---
    counts = list(transitions.values())
    c_max = max(counts) if counts else 1

    edge_colors = assign_edge_colors(transitions)

    # add edges
    for (src, dst), count in transitions.items():
        width = 2 + 6 * (math.log1p(count) / math.log1p(c_max))  # scaled nicely
        net.add_edge(
            src,
            dst,
            width=width,
            label=str(count),
            title=f"{count} vehicles",
            color=edge_colors[(src,dst)],
            arrows="to",
            font={"color": "white", "strokeWidth": 0}
        )

    # physics layout
    net.force_atlas_2based()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=650)


# ---------------- MAIN ----------------
def main():
    st.title("Camera Flow Graph - Night Mode")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("Connection To MinIO failed, bucket does not exist.")
        return

    st.success("Connected to MinIO fileserver successfully.")

    day = st.text_input("Day (YYYY/MM/DD)", "2026/03/27")

    df = load_events(storage, day)

    if df.empty:
        st.warning("No events")
        return

    # ---------------- FILTERS ----------------
    st.sidebar.header("Filters")

    min_score = st.sidebar.slider("Min ReID score", 0.0, 1.0, 0.0)
    max_gap = st.sidebar.number_input("Max time gap (sec)", value=0)
    max_gap = max_gap if max_gap > 0 else None

    cameras = sorted(df["camera_id"].unique())
    selected_cams = st.sidebar.multiselect(
        "Filter cameras",
        cameras,
        default=cameras
    )
    df = df[df["camera_id"].isin(selected_cams)]

    # ---------------- BUILD ----------------
    transitions = build_transitions(df, min_score, max_gap)
    if not transitions:
        st.warning("No transitions after filtering")
        return

    trans_df = transitions_to_df(transitions)

    # ---------------- KPIs ----------------
    col1, col2 = st.columns(2)
    col1.metric("Cameras", len(set(df["camera_id"])))
    col2.metric("Transitions", len(trans_df))

    # ---------------- GRAPH ----------------
    st.subheader("Flow Graph")
    render_graph(transitions)

    # ---------------- TABLE ----------------
    with st.expander("Show transition table"):
        st.dataframe(trans_df, use_container_width=True)


if __name__ == "__main__":
    main()
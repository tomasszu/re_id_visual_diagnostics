import streamlit as st
import pandas as pd
import os, json
from data_loader import StorageBackend
from minio_backend import MinioBackend


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

# ---------------- DISCOVERY ----------------
@st.cache_data
def discover_days(_storage):
    days = set()
    for key in _storage.list_objects("vehicle_events/"):
        parts = key.split("/")
        if len(parts) >= 4:
            days.add(f"{parts[1]}/{parts[2]}/{parts[3]}")
    return sorted(days, reverse=True)

# ---------------- LOAD ----------------
@st.cache_data
def load_events(_storage, day):
    prefix = f"vehicle_events/{day}"
    rows = []

    for key in _storage.list_objects(prefix):
        raw = _storage.get_object(key)
        rows.append(json.loads(raw))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["end_timestamp_utc"], errors="coerce")
    return df.sort_values("datetime")


# ---------------- TRANSITIONS ----------------
def build_transitions(df):
    transitions = {}

    for vid, group in df.groupby("vehicle_id"):
        group = group.sort_values("datetime")

        prev_cam = None

        for row in group.itertuples():
            cam = row.camera_id

            if prev_cam and prev_cam != cam:
                key = (prev_cam, cam)
                transitions[key] = transitions.get(key, 0) + 1

            prev_cam = cam

    return transitions


def transitions_to_df(transitions):
    return pd.DataFrame([
        {"from": k[0], "to": k[1], "count": v}
        for k, v in transitions.items()
    ]).sort_values("count", ascending=False)


# ---------------- MAIN ----------------
def main():
    st.title("Traffic Overview")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
            st.error("Connection To MinIO failed, bucket does not exist.")
            return
    st.success("Connected to MinIO fileserver successfully.")

    # ---------- SELECT DAY ----------
    days = discover_days(storage)
    if not days:
        st.warning("No vehicle events found")
        return

    selected_day = st.selectbox("Select day", days)

    df = load_events(storage, selected_day)

    if df.empty:
        st.warning("No events")
        return

    transitions = build_transitions(df)
    trans_df = transitions_to_df(transitions)

    # --- KPIs ---
    col1, col2, col3 = st.columns(3)

    col1.metric("Unique Vehicles", df["vehicle_id"].nunique())
    col2.metric("Total Events", len(df))
    col3.metric("Transitions", len(trans_df))

    st.subheader("Camera Transitions")
    st.dataframe(trans_df, use_container_width=True)


if __name__ == "__main__":
    main()
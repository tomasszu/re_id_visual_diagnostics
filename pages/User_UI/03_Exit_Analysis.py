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

@st.cache_data
def load_events(_storage, day):
    rows = []
    for key in _storage.list_objects(f"vehicle_events/{day}"):
        rows.append(json.loads(_storage.get_object(key)))

    df = pd.DataFrame(rows)

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["end_timestamp_utc"])

    return df.sort_values("datetime")


def compute_exits(df):
    exits = {}

    for vid, group in df.groupby("vehicle_id"):
        group = group.sort_values("datetime")
        last_cam = group.iloc[-1]["camera_id"]
        exits[last_cam] = exits.get(last_cam, 0) + 1

    return exits


def main():
    st.title("Exit Analysis")

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
        st.warning("No data")
        return

    exit_counts = compute_exits(df)

    exit_df = pd.DataFrame([
        {"camera": cam, "exits": cnt}
        for cam, cnt in exit_counts.items()
    ]).sort_values("exits", ascending=False)

    st.dataframe(exit_df, use_container_width=True)


if __name__ == "__main__":
    main()
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
    day = st.text_input("Day", "2026/03/27")

    df = load_events(storage, day)

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
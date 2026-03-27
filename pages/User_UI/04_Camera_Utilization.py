import streamlit as st
import pandas as pd
import os, json
from dotenv import load_dotenv
from minio_backend import MinioBackend

load_dotenv()


def create_storage():
    return MinioBackend(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        bucket=os.getenv("MINIO_BUCKET"),
        secure=False,
    )


@st.cache_data
def load_events(_storage, day):
    rows = []
    for key in _storage.list_objects(f"vehicle_events/{day}"):
        rows.append(json.loads(_storage.get_object(key)))

    df = pd.DataFrame(rows)

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp_utc"])

    return df


def main():
    st.title("Camera Utilization")

    storage = create_storage()
    day = st.text_input("Day", "2026/03/27")

    df = load_events(storage, day)

    if df.empty:
        st.warning("No data")
        return

    cam_counts = df["camera_id"].value_counts().reset_index()
    cam_counts.columns = ["camera_id", "event_count"]

    st.dataframe(cam_counts, use_container_width=True)


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import os
import io
import base64
import json

from datetime import date
from PIL import Image

from minio_backend import MinioBackend
from data_loader import StorageBackend, load_vehicle_events_day

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


# ---------------- DATA DISCOVERY ----------------
@st.cache_data
def discover_vehicle_event_days(_storage):
    prefix = "vehicle_events/"
    days = set()
    for key in _storage.list_objects(prefix):
        parts = key.split("/")
        if len(parts) >= 4:
            y, m, d = parts[1], parts[2], parts[3]
            days.add(f"{y}-{m}-{d}")
    return sorted(days, reverse=True)


@st.cache_data
def discover_vehicle_event_cameras(_storage, day):
    prefix = f"vehicle_events/{day.replace('-', '/')}"
    cameras = set()
    for key in _storage.list_objects(prefix):
        parts = key.split("/")
        if len(parts) >= 5:
            cameras.add(parts[4])
    return sorted(cameras)


# ---------------- IMAGE LOADING ----------------
@st.cache_data(hash_funcs={MinioBackend: lambda _: None})
def load_image_bytes(_storage, key):
    return _storage.get_object(key)


def load_image_preview_and_meta(storage, key):
    img_bytes = load_image_bytes(storage, key)
    file_size_kb = round(len(img_bytes) / 1024, 2)
    image = Image.open(io.BytesIO(img_bytes))
    width, height = image.size
    image.thumbnail((150, 150))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    preview = f"data:image/png;base64,{base64_img}"
    return preview, width, height, file_size_kb


# ---------------- EVENT SIGHTINGS ----------------
def load_event_sightings(storage, sighting_keys):
    sightings = []
    for key in sighting_keys:
        raw = storage.get_object(key)
        data = json.loads(raw)
        sightings.append(data)
    return sightings


# ---------------- MAIN PAGE ----------------
def main():
    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("Connection To MinIO failed, bucket does not exist.")
        return

    st.success("Connected to MinIO fileserver successfully.")
    st.title("Vehicle Analysed Sighting Inspector v0.1")

    # --- Select day & camera dynamically ---
    available_days = discover_vehicle_event_days(storage)
    if not available_days:
        st.warning("No vehicle events found in the storage.")
        return

    selected_day = st.selectbox("Select day", available_days)

    available_cameras = discover_vehicle_event_cameras(storage, selected_day)
    selected_cameras = st.multiselect(
        "Select cameras",
        available_cameras,
        default=available_cameras
    )

    # --- Load events for selection ---
    events = load_vehicle_events_day(storage, selected_day, selected_cameras)
    if not events:
        st.warning("No events found for the selected day/cameras.")
        return

    events_df = pd.DataFrame(events)
    events_df["start_dt"] = pd.to_datetime(events_df["start_ts"], unit="ns")
    events_df["merged"] = events_df["track_count"] > 1

    # --- Split merged vs single-track events ---
    merged_df = events_df[events_df["merged"]]
    single_df = events_df[~events_df["merged"]]

    # --- Display tables ---
    st.subheader("Merged Vehicle Events")
    merged_table = st.dataframe(
        merged_df[
            ["event_id","camera_id","track_count","sighting_count","duration_sec","plate"]
        ],
        use_container_width=True
    )

    st.subheader("Single Track Events")
    single_table = st.dataframe(
        single_df[
            ["event_id","camera_id","track_count","sighting_count","duration_sec"]
        ],
        use_container_width=True
    )

    # --- Select event ---
    selected_event = None
    merged_selection = st.session_state.get("merged_selection", None)
    single_selection = st.session_state.get("single_selection", None)

    merged_idx = st.number_input("Select merged event row", min_value=0, max_value=len(merged_df)-1, value=0)
    single_idx = st.number_input("Select single event row", min_value=0, max_value=len(single_df)-1, value=0)

    if not merged_df.empty:
        selected_event = merged_df.iloc[merged_idx]
    elif not single_df.empty:
        selected_event = single_df.iloc[single_idx]

    # --- Event inspection ---
    if selected_event is not None:
        st.header(f"Vehicle Event {selected_event['event_id']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Tracks", selected_event["track_count"])
        col2.metric("Sightings", selected_event["sighting_count"])
        col3.metric("Duration (s)", round(selected_event["duration_sec"],2))

        st.write("Tracks:", selected_event["tracks"])

        # Merge Diagnostics
        scores = selected_event.get("track_merge_scores", {})
        if scores:
            st.subheader("Merge Diagnostics")
            score_rows = []
            for track_id, vals in scores.items():
                row = {
                    "track_id": track_id,
                    "time_score": vals.get("time"),
                    "plate_score": vals.get("plate"),
                    "embedding_score": vals.get("embedding"),
                    "total_score": vals.get("total"),
                }
                score_rows.append(row)
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

        # Show sightings by track
        st.subheader("Sightings by Track")
        sightings = load_event_sightings(storage, selected_event["sightings"])
        df = pd.DataFrame(sightings).sort_values("timestamp_ns")
        tracks = sorted(df["track_id"].unique())

        for track_id in tracks:
            track_df = df[df["track_id"] == track_id]
            st.markdown(f"### Track {track_id}")
            cols = st.columns(6)
            for i, row in enumerate(track_df.itertuples()):
                img_bytes = load_image_bytes(storage, row.image_path)
                img = Image.open(io.BytesIO(img_bytes))
                cols[i % 6].image(img, use_container_width=True)


if __name__ == "__main__":
    main()
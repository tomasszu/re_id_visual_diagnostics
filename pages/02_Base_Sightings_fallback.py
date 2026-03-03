import streamlit as st
import pandas as pd

import os
from minio_backend import MinioBackend
from data_loader import StorageBackend, load_sightings_day_index

from PIL import Image
import io

from datetime import date, time


# ENV variables import
from dotenv import load_dotenv
load_dotenv()

START_DATE = date(2026, 2, 12)
DEFAULT_DATE = date(2026, 2, 26)
TODAY = date.today()

def create_storage_from_env() -> StorageBackend:
    # Tailor the values to your MinIO credentials
    return MinioBackend(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        bucket=os.getenv("MINIO_BUCKET"),
        secure=False,
    )

@st.cache_data
def build_dataframe(_storage, day):
    rows = load_sightings_day_index(_storage, day)
    df = pd.DataFrame(rows)
    return df

@st.cache_data
def load_image_bytes(_storage, key):
    return _storage.get_object(key)

def load_image_preview(storage, key):
    img_bytes = load_image_bytes(storage, key)
    image = Image.open(io.BytesIO(img_bytes))
    image.thumbnail((150, 150))  # small preview
    return image

def main():

    # ---------- CONNECTING TO MINIO BUCKET ----------

    storage = create_storage_from_env()

    if not storage.bucket_exists():
        print("Bucket does not exist.")
        st.error("Connection To MinIO failed, bucket does not exist.")
        return
    else:
        print("Bucket exists. Connected to MinIO fileserver.")
        st.success("Connected to MinIO fileserver successfully.")

    # ---------- INTRO ----------

    st.title("Vehicle Sighting Inspector v0.1")
    st.write("This page allows you to inspect vehicle sightings (the atomary units in the dataframe) for any given available date.")

    # ---------- DATE ----------

    selected_date = st.date_input(
        "Select date",
        value=DEFAULT_DATE,
        min_value=START_DATE,
        max_value=TODAY
    )

    day_str = selected_date.strftime("%Y/%m/%d")

    # ---------- LOAD DATAFRAME ----------

    if st.button("Load Day"):
        
        df = build_dataframe(storage, day_str)
        st.session_state["df"] = df

    if "df" in st.session_state:
        try:
            df = st.session_state["df"]

            df = df.sort_values("timestamp_ns")

            st.write("Total sightings:", len(df))

            # ---------- SIDEBAR FILTERS ----------
            with st.expander("Filter sightings", expanded=False):
                st.write("Apply filters to the sightings dataframe. Click 'Load / Apply Filters' to apply the selected filters.")

                # time of day filter
                st.header("Filters")

                time_range = st.slider(
                    "Time of day",
                    value=(time(0, 0), time(23, 59)),
                    format="HH:mm"
                )

                # camera filter

                camera_options = sorted(df["camera_id"].unique())

                selected_cameras = st.multiselect(
                    "Camera ID",
                    options=camera_options,
                    default=camera_options
                )

                # track_id filter

                track_id_filter = st.number_input(
                    "Track ID (optional)",
                    min_value=0,
                    step=1,
                    value=None,
                    placeholder="Enter track ID"
                )
            # ---------- SIDEBAR ACTION ----------
            
            if st.button("Load / Apply Filters"):
                df = build_dataframe(storage, day_str)

                #time filter

                start_time, end_time = time_range

                df["datetime"] = pd.to_datetime(df["timestamp_utc"])
                df["time_only"] = df["datetime"].dt.time

                df = df[
                    (df["time_only"] >= start_time) &
                    (df["time_only"] <= end_time)
                ]

                # camera filter

                df = df[df["camera_id"].isin(selected_cameras)]

                # track filter

                if track_id_filter:
                    df = df[df["track_id"] == track_id_filter]

                # resolute

                st.session_state["filtered_df"] = df
                st.write("Filtered sightings:", len(df))
            


            display_df = df.head(50).copy()

            display_df["preview"] = display_df["image_path"].apply(
                lambda key: load_image_preview(storage, key)
            )
            st.badge("Showing preview for first 50 sightings. Apply more specific filters to narrow down the results.")
            st.data_editor(
                display_df[["preview", "track_id", "camera_id", "timestamp_utc", "sighting_id" ]],
                column_config={
                    "preview": st.column_config.ImageColumn("Preview")
                },
                use_container_width=True,
                hide_index=True
            )

            selected_id = st.selectbox("Select sighting", df["sighting_id"])

            selected_row = df[df["sighting_id"] == selected_id].iloc[0]

            img_bytes = storage.get_object(selected_row["image_path"])
            image = Image.open(io.BytesIO(img_bytes))

            st.image(image, caption=selected_row["sighting_id"])

            st.sidebar.write("Metadata")
            st.sidebar.json({
                "timestamp": selected_row["timestamp_utc"],
                "camera_id": selected_row["camera_id"],
                "track_id": selected_row["track_id"],
                "vehicle_id": selected_row["vehicle_id"],
                "model": selected_row["model_name"],
                "embedding_dim": selected_row["embedding_dim"]
            })

        except Exception as e:
            st.error(f"Error loading data for {day_str}: {str(e)}")
            return



if __name__ == "__main__":
    main()

    
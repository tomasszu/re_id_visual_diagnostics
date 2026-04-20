import streamlit as st
import pandas as pd
import os
import io
import base64

from datetime import date, time
from PIL import Image

from minio_backend import MinioBackend
from data_loader import StorageBackend, load_sightings_day_index


# ---------------- ENV ----------------
START_DATE = date(2026, 2, 12)
DEFAULT_DATE = date(2026, 4, 9)
TODAY = date.today()


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

    for key in _storage.list_objects("sightings/"):
        parts = key.split("/")
        if len(parts) >= 4:
            try:
                y, m, d = int(parts[1]), int(parts[2]), int(parts[3])
                days.add(date(y, m, d))
            except:
                continue

    return sorted(days)


# ---------------- DATA LOADING ----------------
@st.cache_data
def build_dataframe(_storage, day):
    rows = load_sightings_day_index(_storage, day)
    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("timestamp_ns")
        df["datetime"] = pd.to_datetime(df["timestamp_utc"])
        df["time_only"] = df["datetime"].dt.time

    return df


@st.cache_data(hash_funcs={MinioBackend: lambda _: None})
def load_image_bytes(_storage, key):
    return _storage.get_object(key)


def load_image_preview_and_meta(_storage, key):
    img_bytes = load_image_bytes(_storage, key)

    file_size_kb = round(len(img_bytes) / 1024, 2)

    image = Image.open(io.BytesIO(img_bytes))
    width, height = image.size

    image.thumbnail((150, 150))

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    preview = f"data:image/png;base64,{base64_img}"

    return preview, width, height, file_size_kb

def build_preview_row(key, _storage):
    """ key = image path, _storage = MinIO backend """
    preview, w, h, size_kb = load_image_preview_and_meta(_storage, key)
    return pd.Series({
        "preview": preview,
        "resolution": f"{w}x{h}",
        "size_kb": size_kb
    })


# ---------------- FILTER LOGIC ----------------
def apply_filters(base_df, start_time, end_time, selected_cameras, track_id_filter):

    df = base_df.copy()

    # Time filter
    df = df[
        (df["time_only"] >= start_time) &
        (df["time_only"] <= end_time)
    ]

    # Camera filter
    if selected_cameras:
        df = df[df["camera_id"].isin(selected_cameras)]

    # Track filter
    if track_id_filter is not None:
        df = df[df["track_id"] == track_id_filter]

    return df


# ---------------- MAIN ----------------
def main():

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("Connection To MinIO failed, bucket does not exist.")
        return

    st.success("Connected to MinIO fileserver successfully.")

    st.title("Vehicle Sighting Inspector v0.1")

    # ---------- AVAILABLE DAYS ----------
    available_days = discover_days(storage)

    if not available_days:
        st.warning("No vehicle events found")
        return

    min_date = min(available_days)
    max_date = max(available_days)

    selected_date = st.date_input(
        "Select date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    available_days_set = set(available_days)

    if selected_date not in available_days_set:
        st.warning("No data for selected date")
        return

    # ---------- LOAD DAY ----------

    day_str = selected_date.strftime("%Y/%m/%d")

    if st.button("Load Day"):
        base_df = build_dataframe(storage, day_str)
        st.session_state["base_df"] = base_df
        st.session_state["filtered_df"] = base_df  # initialize filtered

    # ---------- REQUIRE DATA ----------
    if "base_df" not in st.session_state or st.session_state["base_df"].empty:
        st.error("Data error while loading specified day.")
        return

    base_df = st.session_state["base_df"]

    st.write("Total sightings:", len(base_df))

    # ---------- FILTER UI ----------
    with st.expander("Filter sightings", expanded=True):

        # Time filter
        time_range = st.slider(
            "Time of day",
            value=(time(0, 0), time(23, 59)),
            format="HH:mm"
        )

        start_time, end_time = time_range

        # Camera filter
        camera_options = sorted(base_df["camera_id"].unique())

        selected_cameras = st.multiselect(
            "Camera ID",
            options=camera_options,
            default=camera_options
        )

        # Track filter (searchable)
        track_options = sorted(base_df["track_id"].unique())

        track_id_filter = st.selectbox(
            "Track ID (optional)",
            options=track_options,
            index=None,
            placeholder="Search or select track ID"
        )

    # ---------- APPLY FILTERS (REACTIVE) ----------
    filtered_df = apply_filters(
        base_df,
        start_time,
        end_time,
        selected_cameras,
        track_id_filter
    )

    st.session_state["filtered_df"] = filtered_df

    if filtered_df.empty:
        st.warning("No sightings match selected filters.")
        return

    # ---------- TABLE PREVIEW ----------
    display_df = filtered_df.head(50).copy()

    preview_meta = display_df["image_path"].apply(build_preview_row, args=(storage,))

    display_df = pd.concat([display_df, preview_meta], axis=1)

    st.space()
    # The table
    st.header("Sightings")
    st.info(f"Showing {len(display_df)} of {len(filtered_df)} filtered sightings")

    table = st.dataframe(
        display_df[[
            "preview",
            "resolution",
            "size_kb",
            "track_id",
            "camera_id",
            "timestamp_utc"
        ]],
        column_config={
            "preview": st.column_config.ImageColumn("Preview"),
            "size_kb": st.column_config.NumberColumn("Size (KB)", format="%.2f")
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    st.space()

    # ---------- ROW SELECTION ----------
    if table.selection.rows:
        selected_idx = table.selection.rows[0]
        selected_row = display_df.iloc[selected_idx]

        img_bytes = load_image_bytes(storage, selected_row["image_path"])
        image = Image.open(io.BytesIO(img_bytes))

        st.image(image, caption=selected_row.get("sighting_id"))

        # drop UI-only fields
        metadata = selected_row.drop(
            ["preview", "resolution", "size_kb", "time_only", "datetime"],
            errors="ignore"
        ).to_dict()

        st.sidebar.header("Metadata")
        st.sidebar.json(metadata)
    else:
        st.info("Select a row in the table to see image and metadata details.")


if __name__ == "__main__":
    main()
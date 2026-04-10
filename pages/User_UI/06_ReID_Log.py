import streamlit as st
import pandas as pd
import os, json, hashlib
import numpy as np
from datetime import datetime
from data_loader import StorageBackend
from minio_backend import MinioBackend
from datetime import datetime, time


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


# ---------------- COLOR ----------------
def vehicle_color(vehicle_id: str):
    h = hashlib.md5(vehicle_id.encode()).hexdigest()
    return f"#{h[:6]}"

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
    rows = []
    prefix = f"vehicle_events/{day}"
    for key in _storage.list_objects(prefix):
        raw = _storage.get_object(key)
        rows.append(json.loads(raw))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["end_timestamp_utc"], errors="coerce")
    df = df.sort_values(["vehicle_id", "datetime"]).reset_index(drop=True)

    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")

    # --- previous camera + gap ---
    df["prev_camera"] = df.groupby("vehicle_id")["camera_id"].shift(1)

    df["delta_sec"] = (
        df["datetime"] -
        df.groupby("vehicle_id")["datetime"].shift(1)
    ).dt.total_seconds()

    df["gap_warning"] = (
        df["delta_sec"].notna() &
        (df["delta_sec"] > 5)
    )

    df["camera_jump"] = (
        df["prev_camera"].notna() &
        (df["camera_id"] != df["prev_camera"])
    )

    # mark new vehicles
    df["is_new"] = df["reid_score"].isna() | (df["reid_score"] == 0)

    # Ensure similarity score exists (if missing, default to 0)
    if "reid_score" not in df.columns:
        df["reid_score"] = np.nan

    return df


# ---------------- MAIN ----------------
def main():
    st.title("Vehicle Event Timeline")

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
        st.warning("No events found")
        return

    # ---------- CAM FILTER ----------
    st.sidebar.header("Filters")
    selected_cams = st.sidebar.multiselect(
        "Filter cameras", sorted(df["camera_id"].unique()), default=None
    )
    if selected_cams:
        df = df[df["camera_id"].isin(selected_cams)]

    # ---------- TIME FILTER ----------
    with st.expander("Filter events", expanded=True):
        time_range = st.slider(
            "Time of day",
            value=(time(0, 0), time(23, 59)),
            format="HH:mm"
        )

    start_time, end_time = time_range

    # Apply time-of-day filter
    df["time_only"] = df["datetime"].dt.time

    if start_time <= end_time:
        # Normal case (e.g. 08:00 → 18:00)
        df = df[
            (df["time_only"] >= start_time) &
            (df["time_only"] <= end_time)
        ]
    else:
        # Overnight case (e.g. 22:00 → 06:00)
        df = df[
            (df["time_only"] >= start_time) |
            (df["time_only"] <= end_time)
        ]

    df = df.sort_values("datetime").reset_index(drop=True)

    st.header("Timeline")

    for row in df.itertuples():
        color = vehicle_color(row.vehicle_id)

        with st.container():
            cols = st.columns([1, 2, 3, 2, 1])

            # Time
            cols[0].markdown(f"**{row.time_str}**")

            # Vehicle ID block with background
            cols[1].markdown(
                f"<span style='background-color:{color};padding:4px 8px;border-radius:6px;color:white;font-weight:bold'>{row.vehicle_id[:8]}</span>",
                unsafe_allow_html=True
            )

            # Camera info
            # if row.is_new:
            #     cols[2].markdown("NEW", unsafe_allow_html=True)
            # else:
                #cols[2].markdown(f"{row.prev_camera} → {row.camera_id}", unsafe_allow_html=True)
            
            cols[2].markdown(f"{row.camera_id}", unsafe_allow_html=True)

            # REID / NEW badge as colored block
            if row.is_new:
                cols[3].markdown(
                    f"<span style='background-color:#e57373;padding:4px 8px;border-radius:6px;color:white;font-weight:bold'>NEW</span>",
                    unsafe_allow_html=True
                )
            else:
                cols[3].markdown(
                    f"<span style='background-color:#81c784;padding:4px 8px;border-radius:6px;color:white;font-weight:bold'>REID {row.reid_score:.2f}</span>",
                    unsafe_allow_html=True
                )

            # Gap
            gap_text = f"{row.delta_sec:.1f}s" if pd.notna(row.delta_sec) else "---"
            if row.gap_warning:
                cols[4].markdown(f"<span style='color:red;font-weight:bold'>{gap_text}</span>", unsafe_allow_html=True)
            else:
                cols[4].markdown(gap_text)


if __name__ == "__main__":
    main()
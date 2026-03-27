import streamlit as st
import pandas as pd
import os, json, hashlib
from datetime import datetime
from dotenv import load_dotenv
from minio_backend import MinioBackend

load_dotenv()


# ---------------- STORAGE ----------------
def create_storage():
    return MinioBackend(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        bucket=os.getenv("MINIO_BUCKET"),
        secure=False,
    )


# ---------------- COLOR ----------------
def vehicle_color(vehicle_id: str):
    h = hashlib.md5(vehicle_id.encode()).hexdigest()
    return f"#{h[:6]}"


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

    df["datetime"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.sort_values("datetime")
    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")

    # --- previous camera + gap ---
    df["prev_camera"] = df.groupby("vehicle_id")["camera_id"].shift(1)
    df["delta_sec"] = (df["datetime"] - df.groupby("vehicle_id")["datetime"].shift(1)).dt.total_seconds()
    df["gap_warning"] = df["delta_sec"] > 5
    df["camera_jump"] = df["camera_id"] != df["prev_camera"]

    # mark new vehicles
    df["is_new"] = df["prev_camera"].isna()

    # Ensure similarity score exists (if missing, default to 0)
    if "reid_score" not in df.columns:
        df["reid_score"] = 0.0

    return df


# ---------------- MAIN ----------------
def main():
    st.title("Vehicle Event Timeline")

    storage = create_storage()
    day = st.text_input("Day (YYYY/MM/DD)", "2026/03/27")
    df = load_events(storage, day)

    if df.empty:
        st.warning("No events found")
        return

    st.sidebar.header("Filters")
    selected_cams = st.sidebar.multiselect(
        "Filter cameras", sorted(df["camera_id"].unique()), default=None
    )
    if selected_cams:
        df = df[df["camera_id"].isin(selected_cams)]

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
            if row.is_new:
                cols[2].markdown("NEW", unsafe_allow_html=True)
            else:
                cols[2].markdown(f"{row.prev_camera} → {row.camera_id}", unsafe_allow_html=True)

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
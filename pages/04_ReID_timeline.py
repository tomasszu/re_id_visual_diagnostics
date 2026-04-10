import streamlit as st
import pandas as pd
import os, io, json, base64, hashlib
from PIL import Image
from data_loader import StorageBackend
from minio_backend import MinioBackend
from datetime import time

GAP_WARNING_THRESHOLD_SEC = 10



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
    prefix = f"vehicle_events/{day}"
    rows = []

    keys = sorted(_storage.list_objects(prefix))

    for key in keys:
        raw = _storage.get_object(key)
        data = json.loads(raw)
        rows.append(data)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["end_timestamp_utc"], errors="coerce")
    df = df.sort_values(["vehicle_id", "datetime"]).reset_index(drop=True)

    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")

    # --- GAP LOGIC ---
    df["prev_time"] = df.groupby("vehicle_id")["datetime"].shift(1)
    df["delta_sec"] = (df["datetime"] - df["prev_time"]).dt.total_seconds()
    df["gap_warning"] = (
        df["prev_time"].notna() &
        (df["delta_sec"] > GAP_WARNING_THRESHOLD_SEC)
    )

    # --- CAMERA JUMP ---
    df["prev_camera"] = df.groupby("vehicle_id")["camera_id"].shift(1)
    df["camera_jump"] = (
        df["prev_camera"].notna() &
        (df["camera_id"] != df["prev_camera"])
    )

    df["is_new"] = df["reid_score"].isna() | (df["reid_score"] == 0)

    df = df.groupby("vehicle_id", group_keys=False).apply(fix_group)

    return df

def fix_group(g):
    if g["is_new"].sum() == 0:
        # fallback: pick earliest finalized or earliest row
        idx = g.index[0]
        g.loc[idx, "is_new"] = True
    elif g["is_new"].sum() > 1:
        # keep only first
        first_idx = g[g["is_new"]].index[0]
        g["is_new"] = False
        g.loc[first_idx, "is_new"] = True
    return g


# ---------------- IMAGE ----------------
@st.cache_data(hash_funcs={MinioBackend: lambda _: None})
def load_image_bytes(_storage, key):
    return _storage.get_object(key)


def preview(_storage, key):
    try:
        img_bytes = load_image_bytes(_storage, key)
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((100, 100))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except:
        return None


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
        st.warning("No events")
        return

    # ---------- SIDEBAR FILTERS ----------
    st.sidebar.header("Filters")

    # --- vehicle id ---
    vehicle_input = st.sidebar.text_input("Vehicle ID (exact match)")

    # --- time of day ---
    time_range = st.sidebar.slider(
        "Time of day",
        value=(time(0, 0), time(23, 59)),
        format="HH:mm"
    )
    start_time, end_time = time_range

    # --- daytime filter ---
    daytime_filter = st.sidebar.selectbox(
        "Daytime filter",
        options=["All", "True", "False"],
        index=0
    )

    # ---------- APPLY FILTERS ----------

    # vehicle id
    if vehicle_input:
        df = df[df["vehicle_id"].str.contains(vehicle_input, case=False, na=False)]

    # time filter
    df["time_only"] = df["datetime"].dt.time

    if start_time <= end_time:
        df = df[
            (df["time_only"] >= start_time) &
            (df["time_only"] <= end_time)
        ]
    else:
        # overnight case
        df = df[
            (df["time_only"] >= start_time) |
            (df["time_only"] <= end_time)
        ]

    # safe fallback
    if "daytime" not in df.columns:
        df["daytime"] = None

    if daytime_filter == "Daytime":
        df = df[df["daytime"] == True]
    elif daytime_filter == "Nighttime":
        df = df[df["daytime"] == False]

    df = df.sort_values("datetime", ascending=True).reset_index(drop=True)

    # --- timeline ---
    for row in df.itertuples():

        rep_img = row.representative["image_path"]
        color = vehicle_color(row.vehicle_id)

        # ---------------- IMAGE HANDLING ----------------
        # use full image for enlarge/lightbox
        full_img_bytes = load_image_bytes(storage, rep_img)
        full_img = Image.open(io.BytesIO(full_img_bytes))

        # only resize separately for thumbnail display
        thumb_img = full_img.copy()
        thumb_img.thumbnail((120, 120))

        col1, col2, col3, col4 = st.columns([1, 2, 3, 2])

        # image:
        # streamlit expands original passed image,
        # therefore pass full_img + width constraint
        col1.image(full_img)

        # time
        col2.markdown(f"**{row.time_str}**")

        # ---------------- SMART TEXT COLOR ----------------
        # choose black/white text depending on brightness
        hex_color = color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "black" if brightness > 150 else "white"

        # vehicle block
        col3.markdown(
            f"""
            <div style='background:{color};
                        padding:8px;
                        border-radius:10px;
                        color:{text_color};
                        font-weight:bold'>
            vehicle: {row.vehicle_id[:8]}<br>
            camera: {row.camera_id}<br>
            track: {row.track_id}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------- SCORE / STATUS ----------------
        is_new = row.is_new

        if is_new:
            col4.error("NEW")
        else:
            col4.success(f"{row.reid_score:.2f}")

        # ---------------- WARNINGS ----------------
        if pd.notna(row.delta_sec) and row.gap_warning:
            st.warning(
                f"Gap >{GAP_WARNING_THRESHOLD_SEC}s: {row.delta_sec:.2f}s (vehicle {row.vehicle_id})"
            )

        if pd.notna(row.prev_camera) and row.camera_jump:
            st.info(
                f"Camera jump: {row.prev_camera} → {row.camera_id} "
                f"(vehicle {row.vehicle_id})"
            )

        st.divider()

    # ---------------- DIAGNOSTICS TABLE ----------------
    st.header("Timeline Diagnostics")

    st.dataframe(
        df[[
            "time_str",
            "vehicle_id",
            "camera_id",
            "track_id",
            "reid_score",
            "delta_sec",
            "gap_warning",
            "camera_jump"
        ]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import os, io, json, base64, hashlib
from PIL import Image
from data_loader import StorageBackend
from minio_backend import MinioBackend
from datetime import time, date

st.set_page_config(layout="wide")

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
            try:
                y, m, d = int(parts[1]), int(parts[2]), int(parts[3])
                days.add(date(y, m, d))
            except:
                continue

    return sorted(days)


# ---------------- LOAD ----------------
@st.cache_data
def load_events(_storage, day):
    enriched_prefix = f"enriched_events/{day}"
    base_prefix = f"vehicle_events/{day}"

    enriched_keys = list(_storage.list_objects(enriched_prefix))

    prefix = enriched_prefix if len(enriched_keys) > 0 else base_prefix

    rows = []

    for key in _storage.list_objects(prefix):
        try:
            raw = _storage.get_object(key)
            data = json.loads(raw)
            rows.append(data)
        except:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["end_timestamp_utc"], errors="coerce")

    # 🔑 IMPORTANT: stable ordering
    df = df.sort_values(["datetime", "vehicle_event_id"]).reset_index(drop=True)

    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")

    # --- GAP ---
    df["prev_time"] = df.groupby("vehicle_id")["datetime"].shift(1)
    df["delta_sec"] = (df["datetime"] - df["prev_time"]).dt.total_seconds()
    df["gap_warning"] = (
        df["prev_time"].notna() &
        (df["delta_sec"] > GAP_WARNING_THRESHOLD_SEC)
    )

    # --- CAMERA ---
    df["prev_camera"] = df.groupby("vehicle_id")["camera_id"].shift(1)
    df["camera_jump"] = (
        df["prev_camera"].notna() &
        (df["camera_id"] != df["prev_camera"])
    )

    # --- NEW FLAG ---
    df["is_new"] = df["reid_score"].isna() | (df["reid_score"] == 0)

    df = df.groupby("vehicle_id", group_keys=False).apply(fix_group)

    return df


def fix_group(g):
    if g["is_new"].sum() == 0:
        g.loc[g.index[0], "is_new"] = True
    elif g["is_new"].sum() > 1:
        first_idx = g[g["is_new"]].index[0]
        g["is_new"] = False
        g.loc[first_idx, "is_new"] = True
    return g


# ---------------- IMAGE ----------------
@st.cache_data(hash_funcs={MinioBackend: lambda _: None})
def load_image_bytes(_storage, key):
    return _storage.get_object(key)


# ---------------- MAIN ----------------
def main():
    st.title("Vehicle Event Timeline")

    storage = create_storage_from_session()

    if not storage.bucket_exists():
        st.error("Connection To MinIO failed")
        return

    st.success("Connected to MinIO")

    # ---------- DATE ----------
    available_days = discover_days(storage)

    if not available_days:
        st.warning("No data")
        return

    selected_date = st.date_input(
        "Select date",
        value=max(available_days),
        min_value=min(available_days),
        max_value=max(available_days)
    )

    if selected_date not in set(available_days):
        st.warning("No data for selected date")
        return

    day_str = selected_date.strftime("%Y/%m/%d")
    df = load_events(storage, day_str)

    if df.empty:
        st.warning("No events")
        return

    # ---------- FILTERS ----------
    st.sidebar.header("Filters")

    vehicle_input = st.sidebar.text_input("Vehicle ID")

    time_range = st.sidebar.slider(
        "Time of day",
        value=(time(0, 0), time(23, 59)),
        format="HH:mm"
    )

    daytime_filter = st.sidebar.selectbox(
        "Daytime",
        ["All", "Daytime", "Nighttime"]
    )

    # apply filters
    if vehicle_input:
        df = df[df["vehicle_id"].str.contains(vehicle_input, case=False, na=False)]

    df["time_only"] = df["datetime"].dt.time

    start_time, end_time = time_range

    if start_time <= end_time:
        df = df[(df["time_only"] >= start_time) & (df["time_only"] <= end_time)]
    else:
        df = df[(df["time_only"] >= start_time) | (df["time_only"] <= end_time)]

    if "daytime" in df.columns:
        if daytime_filter == "Daytime":
            df = df[df["daytime"] == True]
        elif daytime_filter == "Nighttime":
            df = df[df["daytime"] == False]

    # ---------- TIMELINE ----------
    for row in df.itertuples():

        rep_img = row.representative["image_path"]
        color = vehicle_color(row.vehicle_id)

        full_img = Image.open(io.BytesIO(load_image_bytes(storage, rep_img)))

        col1, col2, col3, col4, col5 = st.columns([1, 1.5, 3, 2, 2])

        col1.image(full_img)

        col2.markdown(f"**{row.time_str}**")

        col3.markdown(
            f"""
            <div style='background:{color};padding:8px;border-radius:10px;color:white'>
            vehicle: {row.vehicle_id[:8]}<br>
            camera: {row.camera_id}<br>
            track: {row.track_id}
            </div>
            """,
            unsafe_allow_html=True
        )

        # REID
        if row.is_new:
            col4.error("NEW")
        else:
            col4.success(f"{row.reid_score:.2f}")

        # ---------- LPR BLOCK ----------
        lpr = getattr(row, "LPR", None)

        if lpr:
            plate = lpr.get("plate")
            conf = lpr.get("confidence")

            col5.markdown(
                f"""
                <div style='border-left:4px solid #999;padding-left:8px'>
                <b>Plate:</b> {plate}<br>
                <b>Conf:</b> {conf:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            col5.markdown(
                "<div style='color:#999'>No LPR</div>",
                unsafe_allow_html=True
            )

        # WARNINGS
        if pd.notna(row.delta_sec) and row.gap_warning:
            st.warning(f"Gap {row.delta_sec:.2f}s")

        if pd.notna(row.prev_camera) and row.camera_jump:
            st.info(f"{row.prev_camera} → {row.camera_id}")

        st.divider()

    # ---------- TABLE ----------
    st.header("Diagnostics")

    def extract_plate(x):
        if isinstance(x, dict):
            return x.get("plate")
        return None

    def extract_conf(x):
        if isinstance(x, dict):
            return x.get("confidence")
        return None

    df["plate"] = df.get("LPR").apply(extract_plate) if "LPR" in df else None
    df["lpr_conf"] = df.get("LPR").apply(extract_conf) if "LPR" in df else None

    st.dataframe(
        df[[
            "time_str",
            "vehicle_id",
            "camera_id",
            "track_id",
            "reid_score",
            "plate",
            "lpr_conf",
            "delta_sec",
            "gap_warning",
            "camera_jump"
        ]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
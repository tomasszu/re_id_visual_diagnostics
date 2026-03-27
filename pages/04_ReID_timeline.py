import streamlit as st
import pandas as pd
import os, io, json, base64, hashlib
from PIL import Image
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
    prefix = f"vehicle_events/{day}"
    rows = []

    for key in _storage.list_objects(prefix):
        raw = _storage.get_object(key)
        data = json.loads(raw)
        rows.append(data)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.sort_values("datetime")

    df["time_str"] = df["datetime"].dt.strftime("%H:%M:%S")

    # --- GAP LOGIC ---
    df["prev_time"] = df.groupby("vehicle_id")["datetime"].shift(1)
    df["delta_sec"] = (df["datetime"] - df["prev_time"]).dt.total_seconds()
    df["gap_warning"] = df["delta_sec"] > 5

    # --- CAMERA JUMP ---
    df["prev_camera"] = df.groupby("vehicle_id")["camera_id"].shift(1)
    df["camera_jump"] = df["camera_id"] != df["prev_camera"]

    return df


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

    storage = create_storage()

    day = st.text_input("Day (YYYY/MM/DD)", "2026/03/27")

    df = load_events(storage, day)

    if df.empty:
        st.warning("No events")
        return

    # --- optional filter ---
    selected_vehicle = st.text_input("Filter by vehicle_id (optional)")

    if selected_vehicle:
        df = df[df["vehicle_id"] == selected_vehicle]

    # --- timeline ---
    st.header("Timeline")

    for row in df.itertuples():

        rep_img = row.representative["image_path"]
        color = vehicle_color(row.vehicle_id)

        col1, col2, col3, col4 = st.columns([1,2,3,2])

        # image
        col1.image(preview(storage, rep_img))

        # time
        col2.markdown(f"**{row.time_str}**")

        # vehicle block (colored)
        col3.markdown(
            f"""
            <div style='background:{color};padding:6px;border-radius:6px;color:white'>
            vehicle: {row.vehicle_id[:8]}<br>
            camera: {row.camera_id}<br>
            track: {row.track_id}
            </div>
            """,
            unsafe_allow_html=True
        )

        # score + flags
        if row.reid_score == 0:
            col4.error("NEW")
        else:
            col4.success(f"{row.reid_score:.2f}")

        # --- WARNINGS ---
        if pd.notna(row.delta_sec) and row.gap_warning:
            st.warning(
                f"Gap >5s: {row.delta_sec:.2f}s (vehicle {row.vehicle_id})"
            )

        if pd.notna(row.prev_camera) and row.camera_jump:
            st.info(
                f"Camera jump: {row.prev_camera} → {row.camera_id} (vehicle {row.vehicle_id})"
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
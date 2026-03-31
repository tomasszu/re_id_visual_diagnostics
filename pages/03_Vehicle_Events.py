import streamlit as st
import pandas as pd
import os, io, json, base64

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


# ---------------- DISCOVERY ----------------
@st.cache_data
def discover_days(_storage):
    days = set()
    for key in _storage.list_objects("vehicle_events/"):
        parts = key.split("/")
        if len(parts) >= 4:
            days.add(f"{parts[1]}/{parts[2]}/{parts[3]}")
    return sorted(days, reverse=True)


@st.cache_data
def load_events(_storage, day):
    prefix = f"vehicle_events/{day}"
    events = []

    for key in _storage.list_objects(prefix):
        raw = _storage.get_object(key)
        data = json.loads(raw)
        data["obj_key"] = key
        events.append(data)

    return events


# ---------------- IMAGE ----------------
@st.cache_data(hash_funcs={MinioBackend: lambda _: None})
def load_image_bytes(_storage, key):
    return _storage.get_object(key)


def build_preview(_storage, key):
    try:
        img_bytes = load_image_bytes(_storage, key)
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((120, 120))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except:
        return None


# ---------------- SIGHTINGS ----------------
def load_event_sightings(_storage, sighting_keys):
    rows = []

    for key in sighting_keys:
        full_key = f"sightings/{key}.json"

        try:
            raw = _storage.get_object(full_key)
            rows.append(json.loads(raw))
        except:
            continue

    return rows


# ---------------- MAIN ----------------
def main():
    st.title("Vehicle ReID Inspector")

    storage = create_storage()

    if not storage.bucket_exists():
        st.error("MinIO connection failed")
        return

    # ---------- SELECT DAY ----------
    days = discover_days(storage)
    if not days:
        st.warning("No vehicle events found")
        return

    selected_day = st.selectbox("Select day", days)

    events = load_events(storage, selected_day)
    if not events:
        st.warning("No events")
        return

    df = pd.DataFrame(events)
    df["start_datetime"] = pd.to_datetime(df["start_timestamp_utc"])
    df["end_datetime"] = pd.to_datetime(df["end_timestamp_utc"])

    # ---------- VEHICLE ID VIEW ----------
    st.header("Vehicle IDs")

    vehicle_groups = df.groupby("vehicle_id")

    vehicle_rows = []
    for vid, group in vehicle_groups:
        rep = group.iloc[0]["representative"]["image_path"]

        vehicle_rows.append({
            "vehicle_id": vid,
            "num_events": len(group),
            "preview": build_preview(storage, rep)
        })

    vehicle_df = pd.DataFrame(vehicle_rows)

    vehicle_table = st.dataframe(
        vehicle_df[["preview", "vehicle_id", "num_events"]],
        column_config={"preview": st.column_config.ImageColumn()},
        selection_mode="single-row",
        on_select="rerun",
        use_container_width=True
    )

    # ---------- SELECT VEHICLE ----------
    if not vehicle_table.selection.rows:
        return

    selected_vehicle = vehicle_df.iloc[vehicle_table.selection.rows[0]]["vehicle_id"]

    st.header(f"Vehicle: {selected_vehicle}")

    vehicle_events = df[df["vehicle_id"] == selected_vehicle].copy()
    vehicle_events = vehicle_events.sort_values("start_datetime")

    # ---------- EVENTS TABLE ----------
    event_rows = []
    for _, row in vehicle_events.iterrows():
        rep_img = row["representative"]["image_path"]

        # HH:MM:SS for table display
        time_str = row["start_datetime"].strftime("%H:%M:%S")

        event_rows.append({
            "preview": build_preview(storage, rep_img),
            "track_id": row["track_id"],
            "start": time_str,            
            "camera_id": row["camera_id"],
            "reid_score": round(row["reid_score"], 4),
            "num_sightings": row["num_sightings"],
            "event_id": row["vehicle_event_id"]
            
        })

    events_df = pd.DataFrame(event_rows)

    event_table = st.dataframe(
        events_df,
        column_config={"preview": st.column_config.ImageColumn()},
        selection_mode="single-row",
        on_select="rerun",
        use_container_width=True
    )

    # ---------- SELECT EVENT ----------
    if not event_table.selection.rows:
        return

    selected_event = events_df.iloc[event_table.selection.rows[0]]

    full_event = vehicle_events[
        vehicle_events["vehicle_event_id"] == selected_event["event_id"]
    ].iloc[0]

    st.header(f"Event {selected_event['event_id']}")

    # Show full timestamp in drilldown
    st.metric("Track ID", full_event["track_id"])
    st.metric("ReID Score", round(full_event["reid_score"], 4))
    st.metric("Sightings", full_event["num_sightings"])
    st.write("Time from:", full_event["start_datetime"])
    st.write("Time to:", full_event["end_datetime"])

    # ---------- SIGHTINGS ----------
    st.subheader("Sightings")

    sightings = load_event_sightings(storage, full_event["sightings"])

    if not sightings:
        st.warning("No sightings found")
        return

    sdf = pd.DataFrame(sightings).sort_values("timestamp_ns")

    cols = st.columns(6)
    for i, row in enumerate(sdf.itertuples()):
        try:
            img_bytes = load_image_bytes(storage, row.image_path)
            img = Image.open(io.BytesIO(img_bytes))
            cols[i % 6].image(img, use_container_width=True)
        except:
            continue


if __name__ == "__main__":
    main()
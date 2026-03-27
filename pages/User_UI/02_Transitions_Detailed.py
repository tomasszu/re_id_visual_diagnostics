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

    return df.sort_values("datetime")


def build_transitions(df, min_score=0.0, max_gap=None):
    transitions = {}

    for vid, group in df.groupby("vehicle_id"):
        group = group.sort_values("datetime")

        prev_row = None

        for row in group.itertuples():

            if row.reid_score < min_score:
                continue

            if prev_row is not None:
                gap = (row.datetime - prev_row.datetime).total_seconds()

                if max_gap and gap > max_gap:
                    prev_row = row
                    continue

                if prev_row.camera_id != row.camera_id:
                    key = (prev_row.camera_id, row.camera_id)
                    transitions[key] = transitions.get(key, 0) + 1

            prev_row = row

    return transitions


def main():
    st.title("Detailed Transitions")

    storage = create_storage()
    day = st.text_input("Day", "2026/03/27")

    df = load_events(storage, day)

    if df.empty:
        st.warning("No data")
        return

    min_score = st.slider("Min ReID score", 0.0, 1.0, 0.0)
    max_gap = st.number_input("Max time gap (sec, optional)", value=0)

    max_gap = max_gap if max_gap > 0 else None

    transitions = build_transitions(df, min_score, max_gap)

    trans_df = pd.DataFrame([
        {"from": k[0], "to": k[1], "count": v}
        for k, v in transitions.items()
    ]).sort_values("count", ascending=False)

    st.dataframe(trans_df, use_container_width=True)


if __name__ == "__main__":
    main()
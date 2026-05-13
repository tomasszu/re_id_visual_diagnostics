import json
import pandas as pd
from datetime import date


# ============================================================
# SAFE
# ============================================================
def safe_dict(x):
    return x if isinstance(x, dict) else {}


def extract_gt(row):
    gt = row.get("ground_truth")
    if isinstance(gt, dict):
        return gt.get("gt_vehicle_id")
    return None

# ============================================================
# NORMALIZATION
# ============================================================
def ensure_dict_column(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(safe_dict)
    else:
        df[column_name] = [{} for _ in range(len(df))]


# ============================================================
# DISCOVERY
# ============================================================
def discover_days(storage):
    days = set()

    for key in storage.list_objects("enriched_events/"):
        parts = key.split("/")

        if len(parts) < 4:
            continue

        try:
            days.add(date(int(parts[1]), int(parts[2]), int(parts[3])))
        except:
            continue

    return sorted(days)


# ============================================================
# LOAD EVENTS (STANDARDIZED SCHEMA)
# ============================================================
def load_events(storage, day):
    rows = []

    for key in storage.list_objects(f"enriched_events/{day}"):
        try:
            data = json.loads(storage.get_object(key))
            data["obj_key"] = key
            rows.append(data)
        except:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ---------------- NORMALIZE ----------------
    ensure_dict_column(df, "ground_truth")
    ensure_dict_column(df, "LPR")
    ensure_dict_column(df, "representative")

    # ---------------- TIME ----------------
    df["start_datetime"] = pd.to_datetime(
        df.get("start_timestamp_utc"),
        errors="coerce"
    )

    # ---------------- GT ----------------
    df["gt_vehicle_id"] = df.apply(extract_gt, axis=1)
    df["is_assigned"] = df["gt_vehicle_id"].notna()

    return df

def event_label(row):
    lpr = row.get("LPR") or {}
    plate = lpr.get("plate", "----")

    return (
        f"{row['start_datetime'].strftime('%H:%M:%S')} | "
        f"{plate} | "
        f"{row['vehicle_event_id'][-6:]}"
    )
from abc import ABC, abstractmethod
from typing import Iterable
import json

class StorageBackend(ABC):
    @abstractmethod
    def list_objects(self, prefix: str = "") -> Iterable[str]:
        pass

    @abstractmethod
    def get_object(self, key: str) -> bytes:
        pass

    @abstractmethod
    def bucket_exists(self) -> bool:
        pass


def load_day(storage: StorageBackend, day: str, model: str):
    '''
    Args:
        day - Str in the form of "YYYY/MM/DD"
        model - model version for embeddings in Str
    '''
    prefixes = []
    for folder in ["embeddings", "images", "sightings", "vehicle_events"]:
        if folder != "embeddings":
            prefixes.append(f"{folder}/{day}")
        else:
            prefixes.append(f"{folder}/{model}/{day}")

    for prefix in prefixes:
        for obj_key in storage.list_objects(prefix):
            raw_bytes = storage.get_object(obj_key)

            # parse embedding here
            print(obj_key, len(raw_bytes))


def load_sightings_day_index(storage: StorageBackend, day: str):
    prefix = f"sightings/{day}"
    rows = []

    for obj_key in storage.list_objects(prefix):
        raw = storage.get_object(obj_key)
        data = json.loads(raw)

        data["obj_key"] = obj_key  # keep reference if needed

        rows.append(data)

    return rows

# this load keeps embeddings nested, purposed for multi embedding architecture (for easier analysis, one might want to flatten those)
def load_analysis_day_index(storage: StorageBackend, day: str):
    prefix = f"analysis/{day}"
    rows = []

    for obj_key in storage.list_objects(prefix):
        raw = storage.get_object(obj_key)
        data = json.loads(raw)

        rows.append(data)   # keep everything as-is

    return rows

def load_vehicle_events_day(storage, day, cameras=None):

    prefix = f"vehicle_events/{day.replace('-', '/')}"
    rows = []

    for obj_key in storage.list_objects(prefix):

        camera_id = obj_key.split("/")[4]

        if cameras and camera_id not in cameras:
            continue

        raw = storage.get_object(obj_key)
        data = json.loads(raw)

        row = {
            "event_id": data["event_id"],
            "camera_id": camera_id,

            "start_ts": data["start_ts"],
            "end_ts": data["end_ts"],
            "last_seen_ts": data["last_seen_ts"],

            "sighting_count": data["sighting_count"],
            "track_count": data["track_count"],
            "duration_sec": data["duration_sec"],
            "embedding_variance": data.get("embedding_variance"),

            "tracks": data["tracks"],
            "sightings": data["sightings"],

            "plate": data.get("plate"),
            "plate_confidence": data.get("plate_confidence"),

            "representative_image": data.get("representative_image"),
            "track_merge_scores": data.get("track_merge_scores", {}),

            "obj_key": obj_key
        }

        rows.append(row)

    return rows
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
    for folder in ["embeddings", "images", "sightings"]:
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

        # For now assume 1 embedding per model
        for model_name, emb_info in data["embeddings"].items():
            rows.append({
                "sighting_id": data["sighting_id"],
                "timestamp_utc": data["timestamp_utc"],
                "timestamp_ns": data["timestamp_ns"],
                "camera_id": data["camera_id"],
                "track_id": data["track_id"],
                "vehicle_id": data["vehicle_id"],
                "image_path": data["image_path"],
                "model_name": model_name,
                "embedding_path": emb_info["path"],
                "embedding_dim": emb_info["dim"],
                "embedding_normalized": emb_info["normalized"]
            })

    return rows

def load_analysis_day_index(storage: StorageBackend, day: str):
    prefix = f"analysis/{day}"
    rows = []

    for obj_key in storage.list_objects(prefix):
        raw = storage.get_object(obj_key)
        data = json.loads(raw)

        # For now assume 1 embedding per model
        for model_name, emb_info in data["embeddings"].items():
            rows.append({
                "sighting_id": data["sighting_id"],
                "timestamp_utc": data["timestamp_utc"],
                "timestamp_ns": data["timestamp_ns"],
                "camera_id": data["camera_id"],
                "track_id": data["track_id"],
                "vehicle_id": data["vehicle_id"],
                "image_path": data["image_path"],
                "model_name": model_name,
                "embedding_path": emb_info["path"],
                "embedding_dim": emb_info["dim"],
                "embedding_normalized": emb_info["normalized"],
                "adequate_size": data["adequate_size"],
                "duplicate": data["duplicate"],
                "daytime": data["daytime"]
            })

    return rows
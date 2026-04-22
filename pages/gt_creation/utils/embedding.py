from data_loader import load_embedding


def load_event_embedding(storage, event):
    emb = event.get("embedding")

    if not isinstance(emb, dict) or not emb:
        return None

    try:
        emb_info = next(iter(emb.values()))
        return load_embedding(storage, emb_info)
    except:
        return None


def load_event_embeddings(storage, event):
    emb = event.get("embedding")

    if not isinstance(emb, dict):
        return []

    out = []

    for v in emb.values():
        try:
            e = load_embedding(storage, v)
            if e is not None:
                out.append(e)
        except:
            continue

    return out
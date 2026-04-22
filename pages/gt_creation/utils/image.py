import io
import base64
from PIL import Image


def load_image(storage, key):
    try:
        if not key:
            return None

        img_bytes = storage.get_object(key)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    except:
        return None


def image_to_b64(img, size=(120, 120)):
    if img is None:
        return ""

    try:
        img = img.copy()
        img.thumbnail(size)

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    except:
        return ""
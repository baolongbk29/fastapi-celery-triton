import base64
import io

import cv2
import numpy as np
import yaml
from PIL import Image

with open("worker/data/coco.yaml") as f:
    CLASSES = yaml.load(f, Loader=yaml.FullLoader)["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def convert_base64_to_image(base64_string: str):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    image = Image.open(buffer)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def convert_image_to_base64(image):
    buff = io.BytesIO()
    image = Image.fromarray(np.uint8(image))
    image.save(buff, format="JPEG")
    base64_string = base64.b64encode(buff.getvalue()).decode()
    return base64_string


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)


def preprocessing(original_image):
    [height, width, _] = original_image.shape
    length = max((height, width))
    scale = length / 640

    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    resize = cv2.resize(image, (640, 640))

    img = resize[np.newaxis, :, :, :] / 255.0
    img = img.transpose((0, 3, 1, 2)).astype(np.float32)
    return img, scale

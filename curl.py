import base64
import io

import cv2
import numpy as np
import requests
from PIL import Image

URL = "http://127.0.0.1:9000/inference"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}


def convert_base64_to_image(base64_string: str):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    image = Image.open(buffer)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


if __name__ == "__main__":
    file_id = "test.jpg"
    img = cv2.imread(file_id)
    _, buffer = cv2.imencode(".jpg", img)
    DATA = {
        "img": base64.b64encode(buffer).decode("utf-8"),
    }
    response = requests.post(URL, headers=HEADERS, json=DATA, verify=False)
    print(response.json())

import base64
import io
import time

import cv2
import numpy as np
import tritonclient.http as httpclient
import yaml
from PIL import Image

client = httpclient.InferenceServerClient(url="triton-inference-server:8000")
with open("data/coco.yaml") as f:
    CLASSES = yaml.load(f, Loader=yaml.FullLoader)["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)


def main(img_base64):
    img = base64.b64decode(img_base64)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pre Processing
    preprocess_start = time.time()
    or_copy = original_image.copy()
    [height, width, _] = original_image.shape
    length = max((height, width))
    scale = length / 640  # bbox scaling(후처리)를 위한 값

    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image  # 정사각형을 만들면서 비율을 유지하기 위해 나머지는 black 처리
    resize = cv2.resize(image, (640, 640))

    img = resize[np.newaxis, :, :, :] / 255.0
    img = img.transpose((0, 3, 1, 2)).astype(np.float32)

    inputs = httpclient.InferInput("images", img.shape, datatype="FP32")
    inputs.set_data_from_numpy(img, binary_data=False)  # FP16일때는 binary 필수
    outputs = httpclient.InferRequestedOutput("output0", binary_data=True)
    preprocess_end = time.time()

    # Inference
    res = client.infer(
        model_name="yolov8s", inputs=[inputs], outputs=[outputs]
    ).as_numpy("output0")
    inference_end = time.time()

    # Post Processing
    outputs = np.array([cv2.transpose(res[0].astype(np.float32))])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
            classes_scores
        )
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    postprocess_end = time.time()
    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            or_copy,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )
    visualize_end = time.time()

    process_time = {
        "preprocess": preprocess_end - preprocess_start,
        "inference": inference_end - preprocess_end,
        "postprocess": postprocess_end - inference_end,
        "visualize": visualize_end - postprocess_end,
        "total": visualize_end - preprocess_start,
    }
    _, buffer = cv2.imencode(".jpg", or_copy)

    return base64.b64encode(buffer).decode("utf-8"), result_boxes, process_time


def convert_base64_to_image(base64_string: str):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    image = Image.open(buffer)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def convert_image_to_base64(image):
    buff = io.BytesIO()
    image = Image.fromarray(np.uint8(image))
    image.save(buff, format="JPEG")
    base64_string = base64.b64encode(buff.getvalue()).decode()
    return base64_string


# if __name__=="__main__":

#     image = cv2.imread(r"E:\BaoLong\fastapi-celery-triton\Docker-serving\test.jpg")
#     image_out, results, time = main(convert_image_to_base64(image))
#     print(time)
#     pil_image=convert_base64_to_image(image_out)
#     pil_image.show()

import json
import logging
import logging.config
import os
import time

import cv2
import numpy as np
import tritonclient.http as httpclient
from celery import Task

from worker.helpers import convert_base64_to_image, preprocessing

from .celery import app

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "localhost:8000")

client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)


@app.task
def inference_yolov8(img_base64: str):
    logger.info("Start inference task")
    try:
        original_image = convert_base64_to_image(img_base64)

        # Pre Processing
        or_copy = original_image.copy()
        preprocess_start = time.time()

        img, scale = preprocessing(original_image)
        inputs = httpclient.InferInput("images", img.shape, datatype="FP32")
        inputs.set_data_from_numpy(img, binary_data=False)
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

        process_time = {
            "preprocess": preprocess_end - preprocess_start,
            "inference": inference_end - preprocess_end,
            "postprocess": postprocess_end - inference_end,
            "total": inference_end - preprocess_start,
        }

        return {"results": result_boxes.tolist(), "process_time": process_time}
    except Exception as e:
        logging.critical(e, exc_info=True)
        return {"results": None, "process_time": {}}

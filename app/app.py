import logging
import logging.config
import os

from celery.result import AsyncResult
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.models import ModelPrediction, RequestModel, TaskTicket
from worker.tasks import inference_yolov8

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()
app = FastAPI()


@app.post("/inference", response_model=TaskTicket, status_code=202)
def inference(request: RequestModel):
    logger.info("Wait for Inference")
    try:
        task_id = inference_yolov8.delay(dict(request).get("img"))
    except Exception as e:
        logging.critical(e, exc_info=True)
    return {"task_id": str(task_id), "status": "Processing"}


@app.get(
    "/result/{task_id}",
    response_model=ModelPrediction,
    status_code=200,
    responses={202: {"model": TaskTicket, "description": "Accepted: Not Ready"}},
)
async def get_inference_result(task_id):
    """Fetch result for given task_id"""
    try:
        task = AsyncResult(task_id)
        if not task.ready():
            print(app.url_path_for("inference"))
            return JSONResponse(
                status_code=202,
                content={"task_id": str(task_id), "status": "Processing"},
            )
        result = task.get()
        logger.info(result)
        return {
            "task_id": task_id,
            "image_result": result["image_result"],
            "status": "Success",
            "result": result["results"],
            "process_time": result["process_time"],
        }
    except Exception as e:
        logging.critical(e, exc_info=True)

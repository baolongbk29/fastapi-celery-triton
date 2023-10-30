from pydantic import BaseModel


class RequestModel(BaseModel):
    img: str


class TaskTicket(BaseModel):
    """ID and status for the async tasks"""

    task_id: str
    status: str


class ModelPrediction(BaseModel):
    """Final result"""

    task_id: str
    image_result: str
    status: str
    result: list
    process_time: dict

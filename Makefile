PYTHON=3.8
N_PROC=8
CONDA_CH=defaults conda-forge pytorch
BASENAME=$(shell basename $(CURDIR))
NVCC_USE=$(notdir $(shell which nvcc 2> NULL))


# services
worker:
	# auto-restart for script modifications
	PYTHONPATH=worker
celery -A worker.celery worker -P processes -c $(N_PROC) -l INFO

api:
	PYTHONPATH=app gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:80 --log-level debug app:app

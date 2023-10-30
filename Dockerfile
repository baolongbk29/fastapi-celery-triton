FROM tiangolo/uvicorn-gunicorn:python3.8
RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt /requirements.txt
COPY requirements-dev.txt /requirements-dev.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements-dev.txt
COPY . /app/

# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80", "--log-level", "debug", "app:app"]

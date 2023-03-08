FROM python:3.10.6-buster
COPY head_hunter /head_hunter
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
COPY setup.py setup.py
RUN pip install .
CMD uvicorn head_hunter.api.fast:app --host 0.0.0.0 --port $PORT

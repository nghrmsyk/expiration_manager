FROM python:3.11.5-slim-bullseye AS base

WORKDIR /app

RUN apt update && apt -y upgrade
RUN apt update && apt install -y git
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install torch torchvision
RUN pip install google-cloud-vision
RUN pip install opencv-python
RUN pip install langchain openai

RUN apt-get update && \
    apt-get install -y \
    git \
    g++ \
    cmake \
    libopencv-dev
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#本番環境
FROM base AS production
COPY ./app /app

#開発環境
FROM base AS development

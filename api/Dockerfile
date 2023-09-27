FROM python:3.11.5-slim-bullseye

WORKDIR /app

RUN apt update && apt -y upgrade
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install torch torchvision

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

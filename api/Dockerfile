FROM python:3.11.5-slim-bullseye

WORKDIR /app

RUN apt update && apt -y upgrade
RUN pip install fastapi
RUN pip install uvicorn

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

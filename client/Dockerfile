FROM python:3.11.5-slim-bullseye AS base

WORKDIR /app

RUN apt update && apt -y upgrade

RUN pip install streamlit
RUN pip install streamlit --upgrade
RUN pip install -U pydantic

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]

#本番環境
FROM base AS production
COPY ./app /app

#開発環境
FROM base AS development

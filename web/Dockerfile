FROM python:3.11.5-slim-bullseye

WORKDIR /app

RUN apt update && apt -y upgrade

RUN pip install streamlit

ENTRYPOINT ["streamlit", "run"]

CMD ["src/main.py"]
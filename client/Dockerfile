FROM python:3.11.5-slim-bullseye

WORKDIR /app

RUN apt update && apt -y upgrade

RUN pip install streamlit
RUN pip install streamlit --upgrade
RUN pip install -U pydantic

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
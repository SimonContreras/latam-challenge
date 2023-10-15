# syntax=docker/dockerfile:1.2
FROM python:3.11.4

COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir /challenge
RUN mkdir /data
COPY ./data ./data
COPY ./challenge/*.py ./challenge

EXPOSE 80
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "80"]

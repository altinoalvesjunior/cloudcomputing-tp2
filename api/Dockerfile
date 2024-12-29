FROM python:3.9-slim-bullseye
LABEL authors="altino"

EXPOSE 9101

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

ENV FLASK_APP=app.py
ENV MODEL_PATH=/data/rules.pkl

CMD ["flask", "run", "--host=0.0.0.0", "--port=7000"]
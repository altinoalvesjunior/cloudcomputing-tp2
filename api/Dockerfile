FROM python:3.9-slim-bullseye
LABEL authors="altino"

EXPOSE 7000

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

ENV FLASK_APP=app.py
ENV MODEL_PATH=/app/data/rules.pkl

CMD ["python", "app.py"]
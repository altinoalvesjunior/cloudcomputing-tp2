FROM python:3.9-slim-bullseye
LABEL authors="altino"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rules_generator.py .

COPY datasets /app/datasets/

CMD ["python", "rules_generator.py"]
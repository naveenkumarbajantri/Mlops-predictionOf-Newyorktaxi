# serve/Dockerfile

FROM python:3.9-slim
WORKDIR /app

COPY serve.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9696

CMD ["python", "serve.py"]

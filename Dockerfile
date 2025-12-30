FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
CMD gunicorn -w 1 --threads 2 --timeout 120 -b 0.0.0.0:${PORT} app:app

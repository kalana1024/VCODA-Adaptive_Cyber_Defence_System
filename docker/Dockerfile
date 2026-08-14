FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[training,deep,streaming,api,dashboard,capture]"
EXPOSE 8000 8501
CMD ["vcoda", "serve-api", "--host", "0.0.0.0", "--port", "8000"]

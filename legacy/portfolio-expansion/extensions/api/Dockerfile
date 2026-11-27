FROM python:3.11.15-slim
WORKDIR /app
COPY requirements-portfolio.lock.txt .
RUN pip install --no-cache-dir -r requirements-portfolio.lock.txt
COPY portfolio portfolio
COPY extensions extensions
COPY results/selected_model.json results/selected_model.json
COPY results/artifacts results/artifacts
EXPOSE 8000
CMD ["uvicorn", "extensions.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

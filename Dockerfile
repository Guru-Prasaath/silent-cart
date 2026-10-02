# Silent Cart Retention Console: production image (Render / Databricks Apps / any container host)
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8050
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Build outputs inside the image if they are missing (repo already ships them, so this is a no-op normally)
RUN test -f outputs/key_metrics.json || python run_pipeline.py --skip-notebook --skip-report --skip-deck

EXPOSE 8050
# 1 worker + threads keeps memory under the 512 MB free-plan limit. GROQ_API_KEY is optional:
# without it the "Draft" button uses templates
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120

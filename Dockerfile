FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINAUDIT_WORKER_USER=agent

RUN useradd --create-home --shell /usr/sbin/nologin agent

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml LICENSE README.md DESIGN.md JUDGES.md demo.py judge_adapter.py ./
COPY docs ./docs
COPY audit_env ./audit_env
COPY tests ./tests
COPY data ./data
COPY trajectories ./trajectories

# Restrict closed evaluator files to root (evaluator) only so the unprivileged 'agent' worker cannot read them
RUN find /app/data -type d -name closed -exec chmod 700 {} + && \
    find /app/data -path '*/closed/*' -type f -exec chmod 600 {} +

CMD ["python", "-m", "audit_env.verify"]

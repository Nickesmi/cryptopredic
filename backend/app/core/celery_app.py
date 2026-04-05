import os
from celery import Celery

# Initialize Celery explicitly pointing to Redis for our message broker and result backend
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "cryptopredic_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker"] # Make sure it knows where our tasks are
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Example scheduling - we can run a task every 1 hour to recalculate anomaly spikes
    beat_schedule={
        "recalc-high-potential-hourly": {
            "task": "app.worker.background_evaluate_gems",
            "schedule": 3600.0, # Every 3600 seconds
        }
    }
)

# tasks.py
from celery import shared_task
import subprocess

@shared_task
def train_model_async():
    # Run the model_train.py script
    subprocess.run(["python", "registration/model_train.py"])
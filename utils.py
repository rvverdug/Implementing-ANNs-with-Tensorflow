from datetime import datetime as dt
from pathlib import Path
import tensorflow as tf


def create_summary_writers(config_name):

    current_time = dt.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"logs/{config_name}/{current_time}/train"
    print(f"saving training logs under: {train_log_path}")
    val_log_path = f"logs/{config_name}/{current_time}/val"
    print(f"saving training logs under: {train_log_path}")

    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)

    return train_summary_writer, val_summary_writer


def write_metrics(model, metrics, writer, step, validation=False):
    """Writes metrics state to file."""
    with writer.as_default():
        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}", metric.result(), step=step)
    if validation:
        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])
    else:
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

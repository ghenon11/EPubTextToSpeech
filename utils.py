import sys
import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
import datetime
import threading
from contextlib import contextmanager
from PIL import Image
import re
import config


def ensure_directories(one_directory):
    try:
        if os.path.isdir(one_directory):
            result = os.makedirs(one_directory, exist_ok=True)
        else:
            result = os.makedirs(os.path.basename(
                os.path.dirname(one_directory)), exist_ok=True)
    except Exception as e:
        print(f"Error ensure directories: {e}")
    return result


def init_logging():
    # Configure logging
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s(%(thread)d) - %(levelname)s - %(message)s")
    # Rotate the log file at each startup
    ensure_directories(config.LOG_FILE)
    if os.path.exists(config.LOG_FILE):
        try:
            for i in range(config.LOG_BACKUP_COUNT, 0, -1):
                old_log = f"{config.LOG_FILE}.{i}"
                older_log = f"{config.LOG_FILE}.{i - 1}" if i > 1 else config.LOG_FILE
                if os.path.exists(old_log):
                    # Remove the target log if it already exists
                    os.remove(old_log)
                if os.path.exists(older_log):
                    os.rename(older_log, old_log)
        except Exception as e:
            print(f"Error rotating logs at startup: {e}")

    logging.basicConfig(
        format='%(asctime)s::%(name)s(%(thread)d)::%(levelname)s::%(message)s',
        level=config.LOG_LEVEL,
        handlers=[
            RotatingFileHandler(config.LOG_FILE, maxBytes=50 *
                                1024 * 1024, backupCount=config.LOG_BACKUP_COUNT)
        ]
    )


def wrap_text(text, width):
    words = text.split()
    wrapped_lines = []
    current_line = []

    for word in words:
        if sum(len(w) for w in current_line) + len(current_line) + len(word) <= width:
            current_line.append(word)
        else:
            wrapped_lines.append(' '.join(current_line))
            current_line = [word]

    wrapped_lines.append(' '.join(current_line))
    return '\n'.join(wrapped_lines)


def resize_image(image_path, max_size):
    image = Image.open(image_path)
    image.thumbnail(max_size, Image.LANCZOS)
    return image


def clean_string(input_string):
    # Use regex to keep only alphabetic characters, underscores, or points
    # Replace non-alphanumeric characters with underscores
    return re.sub(r'\W+', '_', input_string)


def add_timestamp_suffix(file_name):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Split the file name and extension
    name, ext = file_name.rsplit('.', 1)
    # Create the new file name with the timestamp suffix
    new_file_name = f"{name}_{timestamp}.{ext}"
    return new_file_name


def add_suffix(file_name, suffix):
    # Split the file name and extension
    name, ext = file_name.rsplit('.', 1)
    # Create the new file name with the timestamp suffix
    new_file_name = f"{name}_{suffix}.{ext}"
    return new_file_name


def get_main_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    if os.path.dirname(sys.argv[0]):
        return os.path.dirname(sys.argv[0])  # name of file
    return os.path.dirname(__file__)


def has_enough_space(folder):
    """Check if there is at least 1GB of free space."""
    if os.path.isfile(folder) or os.path.isdir(folder):
        total, used, free = shutil.disk_usage(folder)
        return free >= config.MIN_FREE_SPACE_BYTES
    return True


class TimeoutLock(object):
    def __init__(self):
        self._lock = threading.RLock()

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking, timeout)

    @contextmanager
    def acquire_timeout(self, timeout):
        result = self._lock.acquire(timeout=timeout)
        yield result
        if result:
            self._lock.release()

    def release(self):
        self._lock.release()

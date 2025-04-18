import logging
import os
import sys
import datetime
import queue
import threading
from logging.handlers import QueueHandler, QueueListener

# Your existing code for setting up folders
base_folder = "experiment_results_m2n/multipele_object"
folder = generate_folder_name(case_num, base_folder)
model_folder = os.path.join(base_folder, folder)
os.makedirs(model_folder, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(model_folder, f'result_{timestamp}.log')

# Set up logging with queue
log_queue = queue.Queue(-1)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler(sys.stdout)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Create a QueueHandler
queue_handler = QueueHandler(log_queue)

# Create a QueueListener
listener = QueueListener(log_queue, file_handler, console_handler)

# Start the listener in a separate thread
listener_thread = threading.Thread(target=listener.start, daemon=True)
listener_thread.start()

# Set the logger to use the QueueHandler
logger.handlers = []
logger.addHandler(queue_handler)

# Modified LoggerWriter class
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.queue = queue.Queue()

    def write(self, message):
        if message != '\n':
            self.queue.put((self.level, message.strip()))

    def flush(self):
        while not self.queue.empty():
            level, message = self.queue.get()
            self.logger.log(level, message)

# Redirect stdout and stderr
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

# Your training code starts here
print("Training is starting...")




# Your training code ends here

# Flush the LoggerWriters to ensure all messages are processed
sys.stdout.flush()
sys.stderr.flush()

# Stop the listener
listener.stop()
listener_thread.join()

print("Training has completed and all logs have been processed.")

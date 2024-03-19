import logging
import uuid
import os


def log_setup(level=logging.INFO):

    # Get the parent directory of the current directory (src)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # Define the logs directory path
    logs_dir = os.path.join(parent_dir, 'logs')

    # Create the logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)

    # generate unique id
    new_uuid = uuid.uuid4()

    # Configure logging to log to a file in the logs directory
    logging.basicConfig(filename=os.path.join(logs_dir, f'logfile_{new_uuid}.log'), level=level,
    format='%(asctime)s - %(levelname)s - %(message)s')
import os
import sys
import logging

LOG_FORMAT = "[%(asctime)s] - %(levelname)s - %(module)s - %(message)s"


def setup_logging(log_dir="logs"):
    """
    Configura o sistema de Logs de forma central
    """

    log_filepath = os.path.join(log_dir, "running_logs.log")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Sistema de logging configurado com sucesso.")

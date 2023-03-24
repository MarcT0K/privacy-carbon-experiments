"""DP training experiment"""

import logging

import colorlog
import tqdm

logger = colorlog.getLogger()


def setup_logger(log_level=logging.DEBUG):
    logger.handlers = []  # Reset handlers
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s %(levelname)s] %(white)s%(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    file_handler = logging.FileHandler("experiment.log")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    logger.addHandler(handler)
    logger.setLevel(log_level)


def experiment():
    setup_logger()
    logger.info("Experiment begins...")

    logger.info("Experiment ends...")


if __name__ == "__main__":
    experiment()

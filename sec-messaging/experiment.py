"""Secure messaging experiment"""

import email
import glob
import os
import logging

import colorlog
import pandas as pd
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


def get_body_from_enron_email(mail):
    """Extract the content from raw Enron email"""
    msg = email.message_from_string(mail)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)


def extract_enron_sent_emails(maildir_directory="../maildir/") -> pd.DataFrame:
    """Extract the emails from the _sent_mail folder of each Enron mailbox."""
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*/_sent_mail/*")

    mail_contents = []
    for mailfile_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        with open(mailfile_path, "r", encoding="utf-8") as mailfile:
            raw_mail = mailfile.read()
            mail_contents.append(get_body_from_enron_email(raw_mail))

    return pd.DataFrame(data={"filename": mails, "mail_body": mail_contents})


def experiment():
    setup_logger()
    logger.info("Experiment begins...")

    logger.info("Experiment ends...")


if __name__ == "__main__":
    experiment()

"""Secure messaging experiment"""

import email
import glob
import os
import logging

import colorlog
import pandas as pd
import pgpy
import tqdm

from codecarbon import OfflineEmissionsTracker
from pgpy.constants import (
    PubKeyAlgorithm,
    KeyFlags,
    HashAlgorithm,
    SymmetricKeyAlgorithm,
    CompressionAlgorithm,
)

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


def track_energy_footprint(
    tracker, experiment_name, experiment_function, *args, **kwargs
):
    begin_emission = tracker.flush()
    begin_energy = tracker._total_energy.kWh

    logger.warning(experiment_name + " begins")
    experiment_function(*args, **kwargs)

    end_emission = tracker.flush()
    end_energy = tracker._total_energy.kWh
    logger.info("Cost summary:")
    logger.info(f"Carbon footprint: {end_emission - begin_emission} KgCO2e")
    logger.info(f"Energy consumption: {end_energy - begin_energy} KWh")
    logger.warning(experiment_name + " ends...")


def generate_keys():
    alice_key = pgpy.PGPKey.new(PubKeyAlgorithm.RSAEncryptOrSign, 4096)
    uid = pgpy.PGPUID.new("Alice", comment="Alice (sender)", email="alice@example.com")
    alice_key.add_uid(
        uid,
        usage={KeyFlags.Sign, KeyFlags.EncryptCommunications, KeyFlags.EncryptStorage},
        hashes=[
            HashAlgorithm.SHA256,
            HashAlgorithm.SHA384,
            HashAlgorithm.SHA512,
            HashAlgorithm.SHA224,
        ],
        ciphers=[
            SymmetricKeyAlgorithm.AES256,
            SymmetricKeyAlgorithm.AES192,
            SymmetricKeyAlgorithm.AES128,
        ],
        compression=[
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.BZ2,
            CompressionAlgorithm.ZIP,
            CompressionAlgorithm.Uncompressed,
        ],
    )

    bob_key = pgpy.PGPKey.new(PubKeyAlgorithm.RSAEncryptOrSign, 4096)
    uid = pgpy.PGPUID.new("Bob", comment="Bob (receiver)", email="bob@example.com")
    bob_key.add_uid(
        uid,
        usage={KeyFlags.Sign, KeyFlags.EncryptCommunications, KeyFlags.EncryptStorage},
        hashes=[
            HashAlgorithm.SHA256,
            HashAlgorithm.SHA384,
            HashAlgorithm.SHA512,
            HashAlgorithm.SHA224,
        ],
        ciphers=[
            SymmetricKeyAlgorithm.AES256,
            SymmetricKeyAlgorithm.AES192,
            SymmetricKeyAlgorithm.AES128,
        ],
        compression=[
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.BZ2,
            CompressionAlgorithm.ZIP,
            CompressionAlgorithm.Uncompressed,
        ],
    )
    return alice_key, bob_key


def sign_all(mails, alice_key, bob_key):
    ...


def sign_and_encrypt_all(mails, alice_key, bob_key):
    ...


def experiment():
    setup_logger()
    logger.error("Experiment begins...")

    logger.warning("Extracting Enron emails")
    mails = extract_enron_sent_emails()

    logger.warning("Generating cryptographic keys")
    alice_key, bob_key = generate_keys()

    tracker = OfflineEmissionsTracker(
        measure_power_secs=5,
        country_iso_code="FRA",
        output_file="raw_emissions.csv",
        log_level="error",
    )
    tracker.start()

    track_energy_footprint(tracker, "Sign", sign_all, mails, alice_key, bob_key)

    track_energy_footprint(
        tracker, "Sign+encrypt", sign_and_encrypt_all, mails, alice_key, bob_key
    )

    tracker.stop()
    logger.error("Experiment ends...")


if __name__ == "__main__":
    experiment()

"""Secure messaging experiment"""

import email
import glob
import os
import logging

import colorlog
import pandas as pd
import gnupg
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

# TODO: add a communication size estimation
# TODO: add Signal (https://github.com/freedomofpress/signal-protocol)?


def setup_logger(log_level=logging.INFO):
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


def extract_enron_sent_emails(maildir_directory="maildir/") -> pd.DataFrame:
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


def generate_keys(key_type):
    # Key size: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf
    if key_type == "RSA":
        key_params = {"key_type": "RSA", "key_length": 3072}
    elif key_type == "ECC":
        key_params = {
            "key_type": "ECDSA",
            "key_curve": "nistp256",
            "subkey_type": "ECDH",
            "subkey_curve": "nistp256",
        }
    else:
        raise NotImplementedError

    if not os.path.exists("temp"):
        os.mkdir("temp")
    gpg = gnupg.GPG(gnupghome="temp")

    alice_key_config = gpg.gen_key_input(
        **key_params,
        name_real="Alice",
        name_comment="Alice (sender)",
        name_email="alice@example.com",
        no_protection=True,
    )
    alice_key = gpg.gen_key(alice_key_config)

    bob_key_config = gpg.gen_key_input(
        **key_params,
        name_real="Bob",
        name_comment="Bob (receiver)",
        name_email="bob@example.com",
        no_protection=True,
    )
    bob_key = gpg.gen_key(bob_key_config)

    return alice_key, bob_key


def sign_all(mails, sender_key, recv_key):
    sender_keyid = None
    for key_dict in sender_key.gpg.list_keys():
        if key_dict["fingerprint"] == sender_key.fingerprint:
            sender_keyid = key_dict["keyid"]
            break
    assert sender_keyid is not None

    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc=f"Sign only", total=len(mails)
    ):
        signed = sender_key.gpg.sign(row_tuple.mail_body, keyid=sender_keyid)
        assert recv_key.gpg.verify(signed.data)


def sign_and_encrypt_all(mails, sender_key, recv_key):
    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc=f"Sign+Encrypt", total=len(mails)
    ):
        enc_msg = sender_key.gpg.encrypt(
            row_tuple.mail_body, [recv_key.fingerprint], sign=sender_key.fingerprint
        )
        decrypted = recv_key.gpg.decrypt(enc_msg.data)
        assert decrypted.data.decode() == row_tuple.mail_body
        assert decrypted.valid  # verified signature


def encrypt_all(mails, sender_key, recv_key):
    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc=f"Encrypt only", total=len(mails)
    ):
        enc_msg = sender_key.gpg.encrypt(row_tuple.mail_body, [recv_key.fingerprint])
        decrypted = recv_key.gpg.decrypt(enc_msg.data)
        assert decrypted.data.decode() == row_tuple.mail_body


def experiment():
    setup_logger()
    logger.error("Experiment begins...")

    logger.warning("Extracting Enron emails")
    mails = extract_enron_sent_emails()
    assert len(mails) == 30109

    logger.warning("Generating cryptographic keys")
    alice_key_rsa, bob_key_rsa = generate_keys("RSA")
    alice_key_ecc, bob_key_ecc = generate_keys("ECC")

    tracker = OfflineEmissionsTracker(
        measure_power_secs=5,
        country_iso_code="FRA",
        output_file="raw_emissions.csv",
        log_level="error",
    )
    tracker.start()

    try:
        track_energy_footprint(
            tracker, "Encrypt RSA", encrypt_all, mails, alice_key_rsa, bob_key_rsa
        )
        track_energy_footprint(
            tracker, "Encrypt ECC", encrypt_all, mails, alice_key_ecc, bob_key_ecc
        )

        track_energy_footprint(
            tracker, "Sign RSA", sign_all, mails, alice_key_rsa, bob_key_rsa
        )
        track_energy_footprint(
            tracker, "Sign ECC", sign_all, mails, alice_key_ecc, bob_key_ecc
        )

        track_energy_footprint(
            tracker,
            "Sign+encrypt",
            sign_and_encrypt_all,
            mails,
            alice_key_rsa,
            bob_key_rsa,
        )
        track_energy_footprint(
            tracker,
            "Sign+encrypt",
            sign_and_encrypt_all,
            mails,
            alice_key_ecc,
            bob_key_ecc,
        )
    except Exception as err:
        logger.error("Error occured: " + str(err))
    except KeyboardInterrupt:
        logger.error("Caught a keyboard interrupt")

    tracker.stop()
    logger.error("Experiment ends...")


if __name__ == "__main__":
    experiment()

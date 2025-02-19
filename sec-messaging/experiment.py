"""Secure messaging experiment"""

import email
import glob
import os
import logging
import time

from csv import DictWriter

import colorlog
import pandas as pd
import gnupg
import pgpy
import tqdm

from codecarbon import OfflineEmissionsTracker
from pgpy.constants import (
    EllipticCurveOID,
    PubKeyAlgorithm,
    KeyFlags,
    HashAlgorithm,
    SymmetricKeyAlgorithm,
    CompressionAlgorithm,
)

logger = colorlog.getLogger()


class Laboratory:
    FIELDNAMES = [
        "Experiment",
        "Duration",
        "Energy",
        "Carbon",
    ]

    def __init__(self, log_level=logging.INFO, experiment_name="experiments"):
        self.tracker = OfflineEmissionsTracker(
            measure_power_secs=1000,
            country_iso_code="FRA",
            output_file="raw_emissions.csv",
            log_level="error",
        )

        self.started = False
        csv_filename = experiment_name + ".csv"
        csv_filename = experiment_name + ".log"

        # SETUP RESULT CSV
        self.filename = csv_filename
        csv_file = open(self.filename, "w", encoding="utf-8")
        writer = DictWriter(csv_file, fieldnames=self.FIELDNAMES)
        writer.writeheader()

        # SETUP LOGGER
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
        file_handler = logging.FileHandler(csv_filename)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)
        logger.addHandler(handler)
        logger.setLevel(log_level)

    def track_energy_footprint(
        self, experiment_name, experiment_function, *args, **kwargs
    ):
        assert self.started

        begin_emission = self.tracker.flush()
        begin_energy = self.tracker._total_energy.kWh
        begin_time = time.time()

        logger.info("%s begins", experiment_name)
        experiment_function(*args, **kwargs)

        end_emission = self.tracker.flush()
        end_energy = self.tracker._total_energy.kWh
        end_time = time.time()

        carbon_diff = end_emission - begin_emission
        energy_diff = end_energy - begin_energy
        time_diff = end_time - begin_time

        logger.info("Cost summary:")
        logger.info("Carbon footprint: %f KgCO2e", carbon_diff)
        logger.info("Energy consumption: %f KWh", energy_diff)
        logger.info("Duration: %f s", time_diff)
        logger.info("%s ends...", experiment_name)

        with open(self.filename, "a", encoding="utf-8") as csv_file:
            writer = DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "Experiment": experiment_name,
                    "Energy": energy_diff,
                    "Carbon": carbon_diff,
                    "Duration": time_diff,
                }
            )

    def __enter__(self):
        logger.info("Experiments begin...")
        self.tracker.start()
        self.started = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tracker.stop()
        self.started = False
        if exc_type is None:
            logger.info("Experiments end...")
        else:  # Exception found
            logger.error("Error during experiments!")


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


def gnupg_generate_keys(key_type):
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


def gnupg_sign_all(mails, sender_key, recv_key):
    sender_keyid = None
    for key_dict in sender_key.gpg.list_keys():
        if key_dict["fingerprint"] == sender_key.fingerprint:
            sender_keyid = key_dict["keyid"]
            break
    assert sender_keyid is not None

    communication_overhead = 0

    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc="Sign only", total=len(mails)
    ):
        signed = sender_key.gpg.sign(row_tuple.mail_body, keyid=sender_keyid)
        communication_overhead += len(signed.data) - len(row_tuple.mail_body.encode())
        assert recv_key.gpg.verify(signed.data)


def gnupg_sign_and_encrypt_all(mails, sender_key, recv_key):
    communication_overhead = 0
    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc="Sign+Encrypt", total=len(mails)
    ):
        enc_msg = sender_key.gpg.encrypt(
            row_tuple.mail_body, [recv_key.fingerprint], sign=sender_key.fingerprint
        )
        decrypted = recv_key.gpg.decrypt(enc_msg.data)
        assert decrypted.data.decode() == row_tuple.mail_body
        assert decrypted.valid  # verified signature
        communication_overhead += len(enc_msg.data) - len(row_tuple.mail_body.encode())


def gnupg_encrypt_all(mails, sender_key, recv_key):
    communication_overhead = 0
    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc="Encrypt only", total=len(mails)
    ):
        enc_msg = sender_key.gpg.encrypt(row_tuple.mail_body, [recv_key.fingerprint])
        decrypted = recv_key.gpg.decrypt(enc_msg.data)
        assert decrypted.data.decode() == row_tuple.mail_body
        communication_overhead += len(enc_msg.data) - len(row_tuple.mail_body.encode())


def pgpy_generate_keys(key_type):
    if key_type == "RSA":
        alice_key = pgpy.PGPKey.new(PubKeyAlgorithm.RSAEncryptOrSign, 3072)
        bob_key = pgpy.PGPKey.new(PubKeyAlgorithm.RSAEncryptOrSign, 3072)
    elif key_type == "ECC":
        alice_key = pgpy.PGPKey.new(PubKeyAlgorithm.ECDSA, EllipticCurveOID.NIST_P256)
        bob_key = pgpy.PGPKey.new(PubKeyAlgorithm.ECDSA, EllipticCurveOID.NIST_P256)
    else:
        raise NotImplementedError

    alice_uid = pgpy.PGPUID.new(
        "Alice", comment="Alice (sender)", email="alice@example.com"
    )
    alice_key.add_uid(
        alice_uid,
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

    bob_uid = pgpy.PGPUID.new("Bob", comment="Bob (receiver)", email="bob@example.com")
    bob_key.add_uid(
        bob_uid,
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


def pgpy_sign_all(mails, sender_key, _recv_key):

    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc="Sign only", total=len(mails)
    ):
        body = pgpy.PGPMessage.new(row_tuple.mail_body)
        sender_key.sign(body)


def pgpy_encrypt_all(mails, sender_key, recv_key):
    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc="Sign+Encrypt", total=len(mails)
    ):
        recv_key.encrypt(row_tuple.mail_body)


def pgpy_sign_and_encrypt_all(mails, sender_key, recv_key):
    for row_tuple in tqdm.tqdm(
        iterable=mails.itertuples(), desc="Sign+Encrypt", total=len(mails)
    ):
        body = pgpy.PGPMessage.new(row_tuple.mail_body)
        body |= sender_key.sign(body)  # Sign and append the signature

        recv_key.encrypt(body)


def experiment():
    logger.warning("Extracting Enron emails")
    mails = extract_enron_sent_emails()
    assert len(mails) == 30109

    with Laboratory() as lab:
        logger.info("Benchmarking GNUPG implementation")
        logger.info("Generating cryptographic keys")
        alice_key_rsa, bob_key_rsa = gnupg_generate_keys("RSA")
        alice_key_ecc, bob_key_ecc = gnupg_generate_keys("ECC")
        lab.track_energy_footprint(
            "GNUPG Encrypt RSA", gnupg_encrypt_all, mails, alice_key_rsa, bob_key_rsa
        )
        lab.track_energy_footprint(
            "GNUPG Encrypt ECC", gnupg_encrypt_all, mails, alice_key_ecc, bob_key_ecc
        )

        lab.track_energy_footprint(
            "GNUPG Sign RSA", gnupg_sign_all, mails, alice_key_rsa, bob_key_rsa
        )
        lab.track_energy_footprint(
            "GNUPG Sign ECC", gnupg_sign_all, mails, alice_key_ecc, bob_key_ecc
        )

        lab.track_energy_footprint(
            "GNUPG Sign+encrypt RSA",
            gnupg_sign_and_encrypt_all,
            mails,
            alice_key_rsa,
            bob_key_rsa,
        )
        lab.track_energy_footprint(
            "GNUPG Sign+encrypt ECC",
            gnupg_sign_and_encrypt_all,
            mails,
            alice_key_ecc,
            bob_key_ecc,
        )

        logger.info("Benchmarking PGPy implementation")
        logger.info("Generating cryptographic keys")
        alice_key_rsa, bob_key_rsa = pgpy_generate_keys("RSA")
        alice_key_ecc, bob_key_ecc = pgpy_generate_keys("ECC")
        lab.track_energy_footprint(
            "PGPy Encrypt RSA", pgpy_encrypt_all, mails, alice_key_rsa, bob_key_rsa
        )
        lab.track_energy_footprint(
            "PGPy Encrypt ECC", pgpy_encrypt_all, mails, alice_key_ecc, bob_key_ecc
        )

        lab.track_energy_footprint(
            "PGPy Sign RSA", pgpy_sign_all, mails, alice_key_rsa, bob_key_rsa
        )
        lab.track_energy_footprint(
            "PGPy Sign ECC", pgpy_sign_all, mails, alice_key_ecc, bob_key_ecc
        )

        lab.track_energy_footprint(
            "PGPy Sign+encrypt RSA",
            pgpy_sign_and_encrypt_all,
            mails,
            alice_key_rsa,
            bob_key_rsa,
        )
        lab.track_energy_footprint(
            "PGPy Sign+encrypt ECC",
            pgpy_sign_and_encrypt_all,
            mails,
            alice_key_ecc,
            bob_key_ecc,
        )


if __name__ == "__main__":
    experiment()

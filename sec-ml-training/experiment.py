"""Secure ML training experiment"""

import logging

from csv import DictWriter

import colorlog

logger = colorlog.getLogger()


class Laboratory:
    FIELDNAMES = ["Experiment", "Energy", "Nb features", "Carbon"]

    def __init__(self, log_level=logging.INFO, csv_filename="results.csv"):
        self.tracker = OfflineEmissionsTracker(
            measure_power_secs=5,
            country_iso_code="FRA",
            output_file="raw_emissions.csv",
            log_level="error",
        )

        self.started = False

        # SETUP RESULT CSV
        self.filename = csv_filename
        with open(csv_filename, "w", encoding="utf-8") as csv_file:
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
        file_handler = logging.FileHandler("experiment.log")
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

        logger.warning(experiment_name + " begins")
        experiment_function(*args, **kwargs)

        end_emission = self.tracker.flush()
        end_energy = self.tracker._total_energy.kWh

        carbon_diff = end_emission - begin_emission
        energy_diff = end_energy - begin_energy

        logger.info("Cost summary:")
        logger.info(f"Carbon footprint: {carbon_diff} KgCO2e")
        logger.info(f"Energy consumption: {energy_diff} KWh")
        logger.warning(experiment_name + " ends...")

        with open(self.filename, "a", encoding="utf-8") as csv_file:
            writer = DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "Experiment": experiment_name,
                    "Energy": energy_diff,
                    "Carbon": carbon_diff,
                }
            )

    def __enter__(self):
        self.tracker.start()
        self.started = True
        return self

    def __exit__(self, *args, **kwargs):
        self.tracker.stop()
        self.started = False


def experiment():
    logger.info("Experiment begins...")

    logger.info("Experiment ends...")


if __name__ == "__main__":
    experiment()

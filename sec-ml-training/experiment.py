"""Secure ML training experiment

TAKING INSPIRATION FROM THE TRAINING EXAMPLE GIVEN BY ZAMA:
https://github.com/zama-ai/concrete-ml/blob/release/1.8.x/docs/advanced_examples/LogisticRegressionTraining.ipynb
"""

import logging
import signal
import time

from csv import DictWriter
from cycler import cycler
from functools import wraps

import colorlog
import matplotlib.pyplot as plt
import pandas as pd

from codecarbon import OfflineEmissionsTracker
from concrete.compiler import check_gpu_available
from concrete.ml.sklearn import (
    SGDClassifier,
    SGDRegressor,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    SGDClassifier as SklearnSGDClassifier,
    SGDRegressor as SklearnSGDRegressor,
)
from sklearn.model_selection import train_test_split

use_gpu_if_available = False
device = "cuda" if use_gpu_if_available and check_gpu_available() else "cpu"

NB_SAMPLES = 400
TEST_SAMPLE_RATE = 0.25
N_ITERATIONS = 15
RANDOM_STATE = 18746

# FIGURE TEMPLATE
params = {
    "font.size": 15,
    "axes.labelsize": 22,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.alpha": 0.5,
    "scatter.marker": "x",
}
plt.style.use("tableau-colorblind10")
plt.rc(
    "axes",
    prop_cycle=(
        plt.rcParams["axes.prop_cycle"]
        + cycler("linestyle", ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"])
    ),
)

texture_1 = {"hatch": "/"}
texture_2 = {"hatch": "."}
texture_3 = {"hatch": "\\"}
texture_4 = {"hatch": "x"}
texture_5 = {"hatch": "o"}
plt.rcParams.update(params)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


def timeout(seconds=7200):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError()

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds)  # used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


class Laboratory:
    FIELDNAMES = [
        "Model",
        "Encrypted",
        "Nb features",
        "Nb samples",
        "Duration",
        "Energy",
        "Carbon",
    ]

    def __init__(self, log_level=logging.DEBUG, experiment_name="experiments"):
        self.tracker = OfflineEmissionsTracker(
            measure_power_secs=1000,
            country_iso_code="NLD",
            output_file="raw_emissions.csv",
            log_level="error",
        )

        self.started = False

        csv_filename = experiment_name + ".csv"
        log_filename = experiment_name + ".log"
        # SETUP RESULT CSV
        self.filename = csv_filename
        with open(csv_filename, "w", encoding="utf-8") as csv_file:
            writer = DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

        # SETUP LOGGER
        self.logger = colorlog.getLogger()
        self.logger.handlers = []  # Reset handlers
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
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
        )
        self.logger.addHandler(file_handler)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    @timeout()
    def track_energy_footprint(
        self, experiment_info, encrypted, experiment_function, *args, **kwargs
    ):
        assert self.started

        experiment_name = ("Encrypted " if encrypted else "") + experiment_info["model"]

        begin_emission = self.tracker.flush()
        begin_energy = self.tracker._total_energy.kWh
        begin_time = time.time()

        self.logger.info("%s begins", experiment_name)
        experiment_function(*args, **kwargs)

        end_emission = self.tracker.flush()
        end_energy = self.tracker._total_energy.kWh
        end_time = time.time()

        carbon_diff = end_emission - begin_emission
        energy_diff = end_energy - begin_energy
        time_diff = end_time - begin_time

        self.logger.debug("Cost summary:")
        self.logger.debug("Carbon footprint: %f KgCO2e", carbon_diff)
        self.logger.debug("Energy consumption: %f KWh", energy_diff)
        self.logger.debug("Duration: %f s", time_diff)
        self.logger.info("%s ends...", experiment_name)

        with open(self.filename, "a", encoding="utf-8") as csv_file:
            writer = DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "Model": experiment_name,
                    "Encrypted": encrypted,
                    "Energy": energy_diff,
                    "Carbon": carbon_diff,
                    "Nb features": experiment_info["n_features"],
                    "Nb samples": experiment_info["n_samples"],
                    "Duration": time_diff,
                }
            )

    def __enter__(self):
        self.logger.info("Experiments begin...")
        self.tracker.start()
        self.started = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tracker.stop()
        self.started = False
        if exc_type is None:
            self.logger.info("Experiments end...")
        else:  # Exception found
            self.logger.error("Error during experiments!")


def varying_nb_features(laboratory):
    model_timeout = {
        SklearnSGDClassifier: False,
        SGDClassifier: False,
        SklearnSGDRegressor: False,
        SGDRegressor: False,
    }

    for nb_features in [3, 4, 5, 6, 7, 8, 9, 10]:
        laboratory.logger.info("NUMBER OF FEATURES: %d", nb_features)
        for ml_task in ["classification", "classification"]:
            if ml_task == "classification":
                X, y = make_classification(
                    n_features=nb_features,
                    random_state=RANDOM_STATE,
                    n_samples=NB_SAMPLES,
                )
                model_class = SklearnSGDClassifier
                encrypted_model_class = SGDClassifier
            elif ml_task == "classification":
                X, y = make_regression(
                    n_features=nb_features,
                    random_state=RANDOM_STATE,
                    n_samples=NB_SAMPLES,
                )
                model_class = SklearnSGDRegressor
                encrypted_model_class = SGDRegressor
            else:
                raise ValueError

            # Retrieve train and test sets:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SAMPLE_RATE, random_state=RANDOM_STATE
            )

            model = model_class(random_state=RANDOM_STATE, max_iter=N_ITERATIONS)
            encrypted_model = encrypted_model_class(
                random_state=RANDOM_STATE,
                max_iter=N_ITERATIONS,
                fit_encrypted=True,
                parameters_range=(-1.0, 1.0),
                verbose=True,
            )

            experiment_info = {
                "model": type(encrypted_model).__name__,
                "n_features": nb_features,
                "n_samples": y_train.shape[0],
            }

            # Train the model on the test set in clear:
            def plaintext_train():
                model.fit(X_train, y_train)

            try:
                if not model_timeout[model_class]:
                    laboratory.track_energy_footprint(
                        experiment_info=experiment_info,
                        encrypted=False,
                        experiment_function=plaintext_train,
                    )
                else:
                    laboratory.logger.warning(
                        "Skipping plaintext experiment (due to a previous timeout)"
                    )
            except TimeoutError:
                laboratory.logger.error("Timeout error...")
                model_timeout[model_class] = True

            # Perform the training in FHE:
            def ciphertext_train():
                encrypted_model.fit(X_train, y_train, fhe="execute", device=device)

            try:
                if not model_timeout[encrypted_model_class]:
                    laboratory.track_energy_footprint(
                        experiment_info=experiment_info,
                        encrypted=True,
                        experiment_function=ciphertext_train,
                    )
                else:
                    laboratory.logger.warning(
                        "Skipping encrypted experiment (due to a previous timeout)"
                    )
            except TimeoutError:
                laboratory.logger.error("Timeout error...")
                model_timeout[encrypted_model_class] = True


def varying_nb_samples(laboratory):
    model_timeout = {
        SklearnSGDClassifier: False,
        SGDClassifier: False,
        SklearnSGDRegressor: False,
        SGDRegressor: False,
    }
    nb_features = 5
    for nb_samples in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        laboratory.logger.info("NUMBER OF SAMPLES: %d", nb_samples)

        for ml_task in ["classification", "regression"]:
            if ml_task == "classification":
                X, y = make_classification(
                    n_features=nb_features,
                    random_state=RANDOM_STATE,
                    n_samples=nb_samples,
                )
                model_class = SklearnSGDClassifier
                encrypted_model_class = SGDClassifier
            elif ml_task == "regression":
                X, y = make_regression(
                    n_features=nb_features,
                    random_state=RANDOM_STATE,
                    n_samples=nb_samples,
                )
                model_class = SklearnSGDRegressor
                encrypted_model_class = SGDRegressor
            else:
                raise ValueError

            # Retrieve train and test sets:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SAMPLE_RATE, random_state=RANDOM_STATE
            )

            model = model_class(random_state=RANDOM_STATE, max_iter=N_ITERATIONS)
            encrypted_model = encrypted_model_class(
                random_state=RANDOM_STATE,
                max_iter=N_ITERATIONS,
                fit_encrypted=True,
                parameters_range=(-1.0, 1.0),
                verbose=True,
            )

            experiment_info = {
                "model": type(encrypted_model).__name__,
                "n_features": nb_features,
                "n_samples": y_train.shape[0],
            }

            # Train the model on the test set in clear:
            def plaintext_train():
                model.fit(X_train, y_train)

            try:
                if not model_timeout[model_class]:
                    laboratory.track_energy_footprint(
                        experiment_info=experiment_info,
                        encrypted=False,
                        experiment_function=plaintext_train,
                    )
                else:
                    laboratory.logger.warning(
                        "Skipping plaintext experiment (due to a previous timeout)"
                    )
            except TimeoutError:
                laboratory.logger.error("Timeout error...")
                model_timeout[model_class] = True

            # Perform the training in FHE:
            def ciphertext_train():
                encrypted_model.fit(X_train, y_train, fhe="execute", device=device)

            try:
                if not model_timeout[encrypted_model_class]:
                    laboratory.track_energy_footprint(
                        experiment_info=experiment_info,
                        encrypted=True,
                        experiment_function=ciphertext_train,
                    )
                else:
                    laboratory.logger.warning(
                        "Skipping encrypted experiment (due to a previous timeout)"
                    )
            except TimeoutError:
                laboratory.logger.error("Timeout error...")
                model_timeout[encrypted_model_class] = True


def draw_figures():
    for variable in ["features", "samples"]:
        results = pd.read_csv(f"varying_nb_{variable}.csv")
        for col_name, label in [
            ("Energy", "Average Energy\nConsumption (kWh)"),
            ("Carbon", "Average Carbon\nFootprint (kg eq.CO2)"),
            ("Duration", "Runtime (s)"),
        ]:
            fig, ax = plt.subplots()

            model = "SGDClassifier"

            model_results = results[results["Model"] == model]
            x = list(model_results[f"Nb {variable}"])
            y = list(model_results[col_name])
            assert len(x) != 0 and len(y) != 0
            ax.plot(x, y, label="In plaintext", marker="x")

            encrypted_model_results = results[results["Model"] == "Encrypted " + model]
            x = list(encrypted_model_results[f"Nb {variable}"])
            y = list(encrypted_model_results[col_name])
            ax.plot(x, y, label="Under encryption", marker="o")

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set(xlabel=f"Number of {variable}", ylabel=label)
            ax.legend(loc="upper left", prop={"size": 12}, framealpha=0.80)
            ax.set_axisbelow(True)
            ax.yaxis.grid(color="gray", linestyle="dashed")
            ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(f"varying_nb_{variable}_{col_name.lower()}.png", dpi=400)


def experiment():
    with Laboratory(experiment_name="varying_nb_features") as lab:
        lab.logger.info("Benchmarking the influence of the number of features")
        varying_nb_features(lab)

    with Laboratory(experiment_name="varying_nb_samples") as lab:
        lab.logger.info("Benchmarking the influence of the number of samples")
        varying_nb_samples(lab)

    draw_figures()


if __name__ == "__main__":
    experiment()

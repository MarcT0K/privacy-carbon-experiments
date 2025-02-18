"""Secure ML inference experiment

TAKING INSPIRATION FROM THE EXAMPLE GIVEN BY ZAMA:
https://docs.zama.ai/concrete-ml/built-in-models/linear#example
"""

import logging
import time

from csv import DictWriter
from functools import partial

import colorlog

from codecarbon import OfflineEmissionsTracker
from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Lasso,
    LinearSVC,
    LinearSVR,
    LinearRegression,
    LogisticRegression,
    NeuralNetClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    XGBClassifier,
    XGBRegressor,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

NB_SAMPLES = 40  # FOR DEBUG: replace with 4000 afterwards
TEST_SAMPLE_RATE = 0.25
RANDOM_STATE = 7568


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
            country_iso_code="FRA",
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


def generic_benchmark_model(laboratory, model_class, task="classification"):
    nb_features = 30

    if task == "classification":
        X, y = make_classification(
            n_features=nb_features,
            random_state=RANDOM_STATE,
            n_samples=NB_SAMPLES,
        )
    elif task == "regression":
        X, y = make_regression(
            n_features=nb_features,
            random_state=RANDOM_STATE,
            n_samples=NB_SAMPLES,
        )
    else:
        raise ValueError("Invalid ML task")

    # Retrieve train and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SAMPLE_RATE, random_state=RANDOM_STATE
    )

    # Instantiate the model:
    model = model_class()

    experiment_info = {
        "model": type(model).__name__,
        "n_features": nb_features,
        "n_samples": y_test.shape[0],
    }

    laboratory.logger.info("Training model %s", experiment_info["model"])

    # Fit the model:
    model.fit(X_train, y_train)

    # Evaluate the model on the test set in clear:
    def plaintext_predict():
        return model.predict(X_test)

    laboratory.track_energy_footprint(
        experiment_info=experiment_info,
        encrypted=False,
        experiment_function=plaintext_predict,
    )

    # Compile the model:
    model.compile(X_train)

    # Perform the inference in FHE:
    def ciphertext_predict():
        return model.predict(X_test, fhe="execute")

    laboratory.track_energy_footprint(
        experiment_info=experiment_info,
        encrypted=True,
        experiment_function=ciphertext_predict,
    )


def varying_nb_features(laboratory):
    neural_net_class = partial(NeuralNetClassifier, module__n_layers=3)

    for nb_features in [5, 10, 20, 40, 50, 75, 100]:
        laboratory.logger.info("NUMBER OF FEATURES: %d", nb_features)
        X, y = make_classification(
            n_features=nb_features,
            random_state=RANDOM_STATE,
            n_samples=NB_SAMPLES,
        )

        # Retrieve train and test sets:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SAMPLE_RATE, random_state=RANDOM_STATE
        )

        for model_class in [
            LogisticRegression,
            RandomForestClassifier,
            neural_net_class,
        ]:
            # Instantiate the model:
            model = model_class()

            experiment_info = {
                "model": type(model).__name__,
                "n_features": nb_features,
                "n_samples": y_test.shape[0],
            }

            laboratory.logger.info("Training model %s", experiment_info["model"])
            # Fit the model:
            model.fit(X_train, y_train)

            # Evaluate the model on the test set in clear:
            def plaintext_predict():
                return model.predict(X_test)

            laboratory.track_energy_footprint(
                experiment_info=experiment_info,
                encrypted=False,
                experiment_function=plaintext_predict,
            )

            # Compile the model:
            model.compile(X_train)

            # Perform the inference in FHE:
            def ciphertext_predict():
                return model.predict(X_test, fhe="execute")

            laboratory.track_energy_footprint(
                experiment_info=experiment_info,
                encrypted=True,
                experiment_function=ciphertext_predict,
            )


def varying_nb_samples(laboratory):
    neural_net_class = partial(NeuralNetClassifier, module__n_layers=3)
    nb_features = 30
    for nb_samples in [40, 100, 240, 500, 1000, 2000, 5000, 10000]:
        X, y = make_classification(
            n_features=nb_features,
            random_state=RANDOM_STATE,
            n_samples=nb_samples,
        )

        # Retrieve train and test sets:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SAMPLE_RATE, random_state=RANDOM_STATE
        )
        laboratory.logger.info("NUMBER OF SAMPLES: %d", y_test.shape[0])

        for model_class in [
            LogisticRegression,
            RandomForestClassifier,
            neural_net_class,
        ]:
            # Instantiate the model:
            model = model_class()

            experiment_info = {
                "model": type(model).__name__,
                "n_features": nb_features,
                "n_samples": y_test.shape[0],
            }

            laboratory.logger.info("Training model %s", experiment_info["model"])
            # Fit the model:
            model.fit(X_train, y_train)

            # Evaluate the model on the test set in clear:
            def plaintext_predict():
                return model.predict(X_test)

            laboratory.track_energy_footprint(
                experiment_info=experiment_info,
                encrypted=False,
                experiment_function=plaintext_predict,
            )

            # Compile the model:
            model.compile(X_train)

            # Perform the inference in FHE:
            def ciphertext_predict():
                return model.predict(X_test, fhe="execute")

            laboratory.track_energy_footprint(
                experiment_info=experiment_info,
                encrypted=True,
                experiment_function=ciphertext_predict,
            )


def draw_figures(): ...  # TODO


def experiment():
    with Laboratory(experiment_name="classification_models") as lab:
        lab.logger.info("Benchmarking the classification models")
        neural_net_class = partial(NeuralNetClassifier, module__n_layers=3)
        classif_models = [
            DecisionTreeClassifier,
            LinearSVC,
            LogisticRegression,
            neural_net_class,
            RandomForestClassifier,
            XGBClassifier,
        ]
        for model_class in classif_models:
            generic_benchmark_model(lab, model_class, task="classification")

    with Laboratory(experiment_name="regression_models") as lab:
        lab.logger.info("Benchmarking the regression models")
        reg_models = [
            DecisionTreeRegressor,
            Lasso,
            LinearSVR,
            LinearRegression,
            RandomForestRegressor,
            Ridge,
            XGBRegressor,
        ]
        for model_class in reg_models:
            generic_benchmark_model(lab, model_class, task="regression")

    with Laboratory(experiment_name="varying_nb_features") as lab:
        lab.logger.info("Benchmarking the influence of the number of features")
        varying_nb_features(lab)

    with Laboratory(experiment_name="varying_nb_samples") as lab:
        lab.logger.info("Benchmarking the influence of the number of samples")
        varying_nb_samples(lab)

    draw_figures()


if __name__ == "__main__":
    experiment()

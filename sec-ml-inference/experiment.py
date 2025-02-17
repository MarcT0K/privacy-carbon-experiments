"""Secure ML inference experiment"""

import logging
import time

from csv import DictWriter
from functools import partial

import colorlog

from codecarbon import OfflineEmissionsTracker
from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    # KNeighborsClassifier,
    Lasso,
    LinearSVC,
    LinearSVR,
    LinearRegression,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    XGBClassifier,
    XGBRegressor,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


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


def benchmark_model(laboratory, model_class, task="classification"):
    nb_features = 30
    nb_samples = 40  # FOR DEBUG: replace with 4000 afterwards
    test_sample_rate = 0.25
    nb_test_samples = nb_samples * test_sample_rate

    if task == "classification":
        X, y = make_classification(
            n_features=nb_features,
            random_state=2,
            n_samples=nb_samples,
        )
    elif task == "regression":
        X, y = make_regression(
            n_features=nb_features,
            random_state=2,
            n_samples=nb_samples,
        )
    else:
        raise ValueError("Invalid ML task")

    # Retrieve train and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sample_rate, random_state=42
    )

    # Instantiate the model:
    model = model_class()

    experiment_info = {
        "model": type(model).__name__,
        "n_features": nb_features,
        "n_samples": nb_test_samples,
    }

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
            # KNeighborsClassifier, # Some parameter issues
            LinearSVC,
            LogisticRegression,
            neural_net_class,
            RandomForestClassifier,
            XGBClassifier,
        ]
        for model_class in classif_models:
            benchmark_model(lab, model_class, task="classification")

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
            benchmark_model(lab, model_class, task="regression")

    with Laboratory(experiment_name="varying_nb_features") as lab:
        lab.logger.info("Benchmarking the influence of the number of features")
        # Logistic regression, NN, RandomForrest
        # TODO

    with Laboratory(experiment_name="varying_nb_bits") as lab:
        lab.logger.info("Benchmarking the influence of the number of bits")
        # Logistic regression, NN, RandomForrest
        # TODO

    draw_figures()


if __name__ == "__main__":
    experiment()

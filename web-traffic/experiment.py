import random
import warnings
import colorlog
import logging
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import requests
from cycler import cycler
from time import sleep
from pathlib import Path
from sys import exit
from codecarbon import OfflineEmissionsTracker
from concurrent.futures import ThreadPoolExecutor

# TERMINAL EXECUTION COMMAND
# sudo /path/to/your/virtualenv/bin/python path/to/your/project_folder/experiment.py

# --------------------------------------------------LOGGER--------------------------------------------------------------#

logger = colorlog.getLogger()
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
file_handler = logging.FileHandler("web-traffic-experiment.log")
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
)
logger.addHandler(file_handler)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# --------------------------------------------------START OF VARIABLES--------------------------------------------------#

# FIGURE TEMPLATE
params = {
    "font.size": 15,
    "axes.labelsize": 18,
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

# PROJECT_PATH (DO NOT CHANGE)
project_path = os.path.dirname(__file__)

# CERTIFICATE PATH
cert_path = "/etc/nginx/ssl/localhost.crt"

# USER DIRECTORY PATH (DO NOT CHANGE)
user = os.environ.get("SUDO_USER")
home_dir = f"/home/{user}"

# SERVER URL
base_url = "localhost"

# UNITS OF EVERY RELEVANT QUANTITY IN RESULTS FILE
QUANTITIES_UNITS = {
    "duration": ("s", "Duration"),
    "emissions": ("kgCO₂eq", "Carbon footprint"),
    "energy_consumed": ("kWh", "Energy consumption"),
}

# RELEVANT FOLDER TO TEST FOR EACH DUMP
dumps_folder_dict = dict()
dumps_folder_dict["Wikipedia"] = (
    f"{home_dir}/HTTPSCarbonExperimentDownloads/wikipedia-simple-html/simple/"
)
dumps_folder_dict["NYTimes"] = (
    f"{home_dir}/HTTPSCarbonExperimentDownloads/nytimes/www.nytimes.com"
)
dumps_folder_dict["MDN"] = (
    f"{home_dir}/HTTPSCarbonExperimentDownloads/mdn_learn/developer.mozilla.org"
)
dumps_folder_dict["Mastodon Blog"] = (
    f"{home_dir}/HTTPSCarbonExperimentDownloads/mastodon_blog/blog.joinmastodon.org"
)
dumps_folder_dict["xkcd"] = f"{home_dir}/HTTPSCarbonExperimentDownloads/xkcd/xkcd.com"

# AMOUNT OF THREADS
NB_THREADS = 8

# NUMBER OF REQUESTS FOR EACH DUMP
NB_REQUESTS = 10000

# SPECIFIES THE CURRENT DUMP THAT IS BEING TESTED (DO NOT CHANGE)
dump_to_test = ""

# CSV HEADERS
RESULTS_HEADER = [
    "protocol",
    "duration",
    "emissions",
    "emissions_rate",
    "energy_consumed",
    "cpu_power",
    "cpu_energy",
    "ram_power",
    "ram_energy",
]


# ---------------------------------------------------END OF VARIABLES---------------------------------------------------#


# --------------------------------------------------START OF FUNCTIONS--------------------------------------------------#


# Removes all the old results files
def remove_old_results():
    # Walk through the directory and its subfolders
    for root, _, files in os.walk(project_path):
        for file in files:
            # Check if the file is a raw_emissions_{dump_to_test}.csv, file_size_data file or fetch_sizes_data file
            if f"raw_emissions_{dump_to_test}" in file:
                # Store the complete file path
                file_path = os.path.join(root, file)
                try:
                    # Remove the file
                    os.remove(file_path)
                except (PermissionError, FileNotFoundError) as e:
                    logger.error(f"Could not delete {file_path}: {e}")


# Grabs NB_REQUESTS random files from the dump_to_test
def get_random_files():
    all_files = []

    # Walk through the directory and its subfolders
    for root, _, files in os.walk(dumps_folder_dict[dump_to_test]):
        for file in files:
            # Add the file to the all_files list
            complete_path = os.path.join(root, file)
            url_path = complete_path.split("HTTPSCarbonExperimentDownloads")[1]
            all_files.append(url_path)

    if len(all_files) == 0:
        raise ValueError("Cannot find files for %s", dump_to_test)

    # If there are fewer files than NB_REQUESTS, we duplicate them
    if len(all_files) < NB_REQUESTS:
        ratio = (NB_REQUESTS // len(all_files)) + 1
        all_files = all_files * ratio

    # Randomly select the specified number of files
    return random.sample(all_files, NB_REQUESTS)


# Extends the random files list, shuffles the list,
# creates a list with the appropriate data-structure for the experiment
def setup():
    result = dict()

    # Repeat list to increase # of fetches
    files_to_fetch = get_random_files()

    # Shuffle the list for randomness
    random.shuffle(files_to_fetch)

    for protocol in ["http", "https"]:
        # Prepare a list of files
        files_to_fetch = [file for file in files_to_fetch]
        # Store the files to be fetched in the result dictionary
        result[f"files_{protocol}"] = files_to_fetch

    return result


# Performs a single fetch request using {protocol}
def fetch(protocol, file):
    # Store the absolute fetch URL
    fetch_url = f"{protocol}://{base_url}{file}"

    if protocol == "http":
        # Fetch the file using HTTP
        response = requests.get(fetch_url)
    else:
        # Fetch the file using HTTPS, using the local certificate
        response = requests.get(fetch_url, verify=cert_path)

    if response.status_code != 200:
        logger.error("Invalid status code %d for %s", response.status_code, fetch_url)

    return response


# Performs the main experiment
def main_experiment():
    # Suppress warnings from pandas used in CodeCarbon as they are irrelevant
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Call the setup function
    setup_dict = setup()

    # Create a CodeCarbon offline tracker
    tracker = OfflineEmissionsTracker(
        measure_power_secs=1000,
        country_iso_code="NLD",
        output_file=f"{project_path}/raw_emissions_{dump_to_test}.csv",
        log_level="error",
    )

    # Run the experiment for both HTTP and HTTPS
    for protocol in ["http", "https"]:
        try:
            # Store the files to be fetched
            files = setup_dict.get(f"files_{protocol}")

            # Start the tracker before fetching
            tracker.start()

            # Perform the fetches using a thread pool
            with ThreadPoolExecutor(max_workers=NB_THREADS) as executor:
                # Fetch each file using the specified protocol
                executor.map(lambda file: fetch(protocol, file), files)

            # Stop the tracker after the fetches are complete
            tracker.stop()
        except KeyboardInterrupt:
            logger.error("Experiment interrupted by user...")
            raise


# Fetches the results from raw_emissions_{dump_to_test}.csv and generates a results csv file
def gather_results():
    # Fetch the results
    df = pd.read_csv(f"{project_path}/raw_emissions_{dump_to_test}.csv")

    # Grab header row and separate http/https results
    header = df.columns

    # Consistency check
    assert df.shape[0] == 2  # One row for HTTP and one for HTTPS

    df_http = pd.DataFrame(df.iloc[0:1].values, columns=header)
    df_https = pd.DataFrame(df.iloc[1:2].values, columns=header)

    # Open the relevant results file
    with open(f"{project_path}/results_{dump_to_test}.csv", "w", newline="") as file:
        # Create a writer for the file
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # Write the header row
        writer.writerow(RESULTS_HEADER)
        # Write the HTTP values row
        writer.writerow(
            [
                "HTTP",
                df_http["duration"].iloc[0],
                df_http["emissions"].iloc[0],
                df_http["emissions_rate"].iloc[0],
                df_http["energy_consumed"].iloc[0],
                df_http["cpu_power"].iloc[0],
                df_http["cpu_energy"].iloc[0],
                df_http["ram_power"].iloc[0],
                df_http["ram_energy"].iloc[0],
            ]
        )
        # Write the HTTPS values row
        writer.writerow(
            [
                "HTTPS",
                df_https["duration"].iloc[0],
                df_https["emissions"].iloc[0],
                df_https["emissions_rate"].iloc[0],
                df_https["energy_consumed"].iloc[0],
                df_https["cpu_power"].iloc[0],
                df_https["cpu_energy"].iloc[0],
                df_https["ram_power"].iloc[0],
                df_https["ram_energy"].iloc[0],
            ]
        )


# Generates bar plots showcasing a result metric per dump for all dumps combined
def generate_bar_plots():
    # Define the files
    files = []
    for dump in dumps_folder_dict.keys():
        # Store the absolute file path
        file_path = f"{project_path}/results_{dump}.csv"

        # Check if the file exists
        if os.path.exists(file_path):
            # Add the file to the files list
            files.append(file_path)

    # Grab the quantities to plot
    if len(files) == 0:
        logger.error(
            "No results files available. First create results before generating plots..."
        )
        return

    # Iterate over each quantity and generate the plots
    for quantity, (quantity_unit, quantity_name) in QUANTITIES_UNITS.items():
        # Create lists for the labels, HTTP values, HTTPS values, and variances
        x = []
        y_http = []
        y_https = []

        # Extract data for each quantity
        for file in files:
            df = pd.read_csv(file)
            assert df.shape[0] == 2  # 1 line for HTTP and 1 for HTTPS

            http_row = df[df["protocol"] == "HTTP"]
            https_row = df[df["protocol"] == "HTTPS"]

            # Calculate mean and variance for HTTP and HTTPS
            http_mean = http_row[quantity].iloc[0] / NB_REQUESTS
            https_mean = https_row[quantity].iloc[0] / NB_REQUESTS

            # Use the file name (without extension) as the label
            test_name = file.replace(f"{project_path}/results_", "").replace(".csv", "")

            # Add the relevant values to the lists
            x.append(test_name)
            y_http.append(http_mean)
            y_https.append(https_mean)

        # Plot grouped bar chart for the current quantity
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar width and index adjustment
        bar_width = 0.35  # Width of each bar
        index = range(len(x))

        # Create bars for HTTP and HTTPS (with error bars for variances)
        ax.bar(
            index,
            y_http,
            bar_width,
            label=f"HTTP {quantity_name}",
            **texture_1,
        )
        ax.bar(
            [i + bar_width for i in index],
            y_https,
            bar_width,
            label=f"HTTPS {quantity_name}",
            **texture_2,
        )

        # Add labels, title, and legend with increased font sizes
        ax.set_xlabel("Websites")
        ax.set_ylabel(f"Average {quantity_name} ({quantity_unit})")
        ax.set_xticks(
            [i + bar_width / 2 for i in index]
        )  # Adjust x-ticks to be between bars
        ax.set_xticklabels(
            x, rotation=45, ha="right"
        )  # Set x-axis labels with rotation for readability

        ax.legend()

        # Delete the plot if it already exists
        plot_path = f"{project_path}/{quantity}_plot.png"
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # Save the new plot
        plt.savefig(plot_path, format="png", bbox_inches="tight")


# ---------------------------------------------------END OF FUNCTIONS---------------------------------------------------#


# ---------------------------------------------------START OF SCRIPT----------------------------------------------------#

# WARNING: THIS EXPERIMENT REQUIRES ROOT (SUDO)
if __name__ == "__main__":
    # Check if running with root
    if os.geteuid() != 0:
        # Experiment is started without root
        logger.error(
            "Root privileges were not granted. \nThis script needs root in order to generate valid results!"
        )
        # Abort the experiment
        exit(1)

    # Start the experiment with root privileges
    logger.info(f"{os.linesep}Starting the main experiment...")
    # Iterate through every dump
    for dump in dumps_folder_dict.keys():
        # Set the dump_to_test to the dump to test
        dump_to_test = dump

        remove_old_results()

        logger.info(
            f"{os.linesep}{os.linesep}Starting experiment on the {dump_to_test} dump..."
        )
        main_experiment()
        logger.info(f"Experiment {dump_to_test} finished!")

        # Wait until raw_emissions_{dump}.csv is generated (if needed)
        logger.info(
            f"{os.linesep}Waiting for raw_emissions_{dump_to_test}.csv to be generated..."
        )
        counter = 0
        while (
            not Path(f"{project_path}/raw_emissions_{dump_to_test}.csv").exists()
            and counter < 50
        ):
            sleep(0.1)
            counter += 1

        # Put the relevant results into the results_{dump_to_test}.csv file
        logger.info(f"{os.linesep}Writing relevant results to results_{dump_to_test}.csv...")
        gather_results()

    # Wait until every results file is generated
    logger.info(
        f"{os.linesep}{os.linesep}All runs finished! Waiting for all results files to be generated..."
    )
    for dump in dumps_folder_dict.keys():
        while True:
            if Path(f"{project_path}/results_{dump}.csv").exists():
                break
            else:
                sleep(0.1)

    # Generate plots of every column of relevant results for every dump
    logger.info(
        f"{os.linesep}Generating combined bar plots for every individual variable in results files..."
    )
    generate_bar_plots()

    # End of the experiment script
    logger.info(
        f"{os.linesep}All experiments finished!{os.linesep}Terminating...{os.linesep}"
    )

# ----------------------------------------------------END OF SCRIPT-----------------------------------------------------#

import itertools
import random
import math
import warnings
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
# user = os.environ.get("SUDO_USER")
home_dir = os.path.expanduser("~")

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
dumps_folder_dict["Wikipedia"] = f"{home_dir}/Downloads/wikipedia-simple-html/simple/"
dumps_folder_dict["NYTimes"] = f"{home_dir}/Downloads/nytimes/www.nytimes.com"
dumps_folder_dict["MDN"] = f"{home_dir}/Downloads/mdn_learn/developer.mozilla.org"

# AMOUNT OF THREADS
NB_THREADS = 8

# NUMBER OF RANDOM FILES USED FROM DUMP (i.e., NUMBER OF FILES FETCHED PER RUN)
NB_FILES = 1000

# NUMBER OF RUNS PER DUMP
NB_RUNS = 1


# SPECIFIES THE CURRENT DUMP THAT IS BEING TESTED (DO NOT CHANGE)
dump_to_test = ""

# CSV HEADERS
RESULTS_HEADER = [
    "",
    "duration",
    "emissions",
    "emissions_rate",
    "energy_consumed",
    "cpu_power",
    "cpu_energy",
    "ram_power",
    "ram_energy",
    "emissions_variance",
    "emissions_rate_variance",
    "energy_consumed_variance",
]

RATIOS_HEADER = [
    "",
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
            if (
                f"raw_emissions_{dump_to_test}" in file
                or "file_size_data" in file
                or "fetch_sizes_data" in file
            ):
                # Store the complete file path
                file_path = os.path.join(root, file)

                try:
                    # Remove the file
                    os.remove(file_path)
                except (PermissionError, FileNotFoundError) as e:
                    print(f"Could not delete {file_path}: {e}")


# Grabs NB_FILES random files from the dump_to_test
def get_random_files():
    all_files = []

    # Walk through the directory and its subfolders
    for root, _, files in os.walk(dumps_folder_dict[dump_to_test]):
        for file in files:
            # Add the file to the all_files list
            all_files.append(os.path.join(root, file))

    if len(all_files) == 0:
        raise ValueError

    # If there are fewer files than NB_FILES, we duplicate them
    if len(all_files) < NB_FILES:
        ratio = (NB_FILES // len(all_files)) + 1
        all_files = all_files * ratio

    # Randomly select the specified number of files
    return random.sample(all_files, NB_FILES)


# Extends the random files list, shuffles the list,
# creates a list with the appropriate data-structure for the experiment
def setup():
    result = dict()

    # Repeat list to increase # of fetches
    files_to_fetch = get_random_files()
    print("Files to fetch:", files_to_fetch)
    # Shuffle the list for randomness
    random.shuffle(files_to_fetch)

    for protocol in ["http", "https"]:
        # Prepare a list of (protocol, file) tuples
        files_to_fetch = [(protocol, file) for file in files_to_fetch]
        # Store the files to be fetched in the result dictionary
        result[f"files_{protocol}"] = files_to_fetch

    return result


# Performs a single fetch request using {protocol}
def fetch(protocol, file):
    # Store the absolute fetch URL
    fetch_url = f"{protocol}://{base_url}/{file}"

    if protocol == "http":
        # Fetch the file using HTTP
        response = requests.get(fetch_url)
    else:
        # Fetch the file using HTTPS, using the local certificate
        response = requests.get(fetch_url, verify=cert_path)

    return response


# Performs the main experiment
def main_experiment():
    # Suppress warnings from pandas used in CodeCarbon as they are irrelevant
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Call the setup function
    setup_dict = setup()

    # Create a CodeCarbon offline tracker
    tracker = OfflineEmissionsTracker(
        measure_power_secs=1,
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

        except Exception as err:
            print("Error occurred:", err)
        except KeyboardInterrupt:
            print("Experiment interrupted by user...")


# Fetches the results from raw_emissions_{dump_to_test}.csv and generates both a results and ratios csv file
def gather_results():
    # Fetch the results
    df = pd.read_csv(f"{project_path}/raw_emissions_{dump_to_test}.csv")

    # Grab header row and separate http/https results
    header = df.columns
    df_http = pd.DataFrame(df.iloc[::2].values, columns=header)
    df_https = pd.DataFrame(df.iloc[1::2].values, columns=header)

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
                df_http["duration"].mean(),
                df_http["emissions"].mean(),
                df_http["emissions_rate"].mean(),
                df_http["energy_consumed"].mean(),
                df_http["cpu_power"].mean(),
                df_http["cpu_energy"].mean(),
                df_http["ram_power"].mean(),
                df_http["ram_energy"].mean(),
                df_http["emissions"].var(),
                df_http["emissions_rate"].var(),
                df_http["energy_consumed"].var(),
            ]
        )
        # Write the HTTPS values row
        writer.writerow(
            [
                "HTTPS",
                df_https["duration"].mean(),
                df_https["emissions"].mean(),
                df_https["emissions_rate"].mean(),
                df_https["energy_consumed"].mean(),
                df_https["cpu_power"].mean(),
                df_https["cpu_energy"].mean(),
                df_https["ram_power"].mean(),
                df_https["ram_energy"].mean(),
                df_https["emissions"].var(),
                df_https["emissions_rate"].var(),
                df_https["energy_consumed"].var(),
            ]
        )

    # Open the relative ratio (HTTPS vs HTTP) file
    with open(f"{project_path}/ratios_{dump_to_test}.csv", "w", newline="") as file:
        # Create a writer for the file
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # Write the header row
        writer.writerow(RATIOS_HEADER)
        # Write the relative ratios row
        writer.writerow(
            [
                "Relative Ratio (HTTPS vs HTTP)",
                df_https["duration"].mean() / df_http["duration"].mean(),
                df_https["emissions"].mean() / df_http["emissions"].mean(),
                df_https["emissions_rate"].mean() / df_http["emissions_rate"].mean(),
                df_https["energy_consumed"].mean() / df_http["energy_consumed"].mean(),
                df_https["cpu_power"].mean() / df_http["cpu_power"].mean(),
                df_https["cpu_energy"].mean() / df_http["cpu_energy"].mean(),
                df_https["ram_power"].mean() / df_http["ram_power"].mean(),
                df_https["ram_energy"].mean() / df_http["ram_energy"].mean(),
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
        print(
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
            with open(file, "r") as existingFile:
                reader = csv.reader(existingFile, quoting=csv.QUOTE_MINIMAL)
                # Skip the header
                header = next(reader, None)

                # Initialize variables for the quantity data
                http_values = []
                https_values = []

                # Collect HTTP and HTTPS values for the current quantity
                for row in reader:
                    if row[0] == "HTTP":
                        http_values.append(float(row[RESULTS_HEADER.index(quantity)]))
                    elif row[0] == "HTTPS":
                        https_values.append(float(row[RESULTS_HEADER.index(quantity)]))

                # Only append if both HTTP and HTTPS data are found for the file
                if http_values and https_values:
                    # Calculate mean and variance for HTTP and HTTPS
                    http_mean = sum(http_values) / len(http_values)
                    https_mean = sum(https_values) / len(https_values)

                    # Use the file name (without extension) as the label
                    test_name = file.replace(f"{project_path}/results_", "").replace(
                        ".csv", ""
                    )

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

        # Move the legend to the lower right with increased font size
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, -0.25),  # Adjust legend position to align with the title
            borderaxespad=0.1,
        )

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
        print(
            "Root privileges were not granted. \nThis script needs root in order to generate valid results!"
        )
        # Abort the experiment
        exit(1)

    # Start the experiment with root privileges
    print(f"{os.linesep}Starting the main experiment...")
    # Iterate through every dump
    for dump in dumps:
        # Set the dump_to_test to the dump to test
        dump_to_test = dump

        print(
            f"{os.linesep}{os.linesep}Starting {NB_RUNS} runs of the {dump_to_test} dump..."
        )
        # Perform NB_RUNS runs
        for i in range(NB_RUNS):
            main_experiment()
            print(f"Run {i} finished!")

        # Wait until raw_emissions_{dump}.csv is generated (if needed)
        print(
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
        print(f"{os.linesep}Writing relevant results to results_{dump_to_test}.csv...")
        gather_results()

    # Wait until every results file is generated
    print(
        f"{os.linesep}{os.linesep}All runs finished! Waiting for all results files to be generated..."
    )
    for dump in dumps:
        while True:
            if Path(f"{project_path}/results_{dump}.csv").exists():
                break
            else:
                sleep(0.1)

    # Generate plots of every column of relevant results for every dump
    print(
        f"{os.linesep}Generating combined bar plots for every individual variable in results files..."
    )
    generate_bar_plots()

    # End of the experiment script
    print(
        f"{os.linesep}All experiments finished!{os.linesep}Terminating...{os.linesep}"
    )

# ----------------------------------------------------END OF SCRIPT-----------------------------------------------------#

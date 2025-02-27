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

# PROJECT_PATH (DO NOT CHANGE)
project_path = os.path.dirname(__file__)

# CERTIFICATE PATH
cert_path = "/etc/nginx/ssl/localhost.crt"

# USER DIRECTORY PATH (DO NOT CHANGE)
user = os.environ.get("SUDO_USER")
home_dir = os.path.expanduser(f"~{user}")

# SERVER URL
base_url = "localhost"

# LIST OF ALL DUMPS
dumps = ["wikipedia", "nytimes", "github", "mdn_learn", "amazon"]

# UNITS OF EVERY RELEVANT QUANTITY IN RESULTS FILE
QUANTITIES_UNITS = {
    "duration": ("s", "Duration"),
    "emissions": ("kgCO₂eq", "Carbon\nfootprint"),
    "energy_consumed": ("kWh", "Energy\nconsumption"),
}

# RELEVANT FOLDER TO TEST FOR EACH DUMP
dumps_folder_dict = dict()
dumps_folder_dict["wikipedia"] = f"{home_dir}/Downloads/wikipedia-simple-html/simple/"
dumps_folder_dict["nytimes"] = f"{home_dir}/Downloads/nytimes/www.nytimes.com"
dumps_folder_dict["github"] = f"{home_dir}/Downloads/github/github.com"
dumps_folder_dict["mdn_learn"] = f"{home_dir}/Downloads/mdn_learn/developer.mozilla.org"
dumps_folder_dict["amazon"] = f"{home_dir}/Downloads/amazon/www.amazon.nl"

# AMOUNT OF THREADS
threads = 8

# NUMBER OF RANDOM FILES USED FROM DUMP
num_files = 50

# NUMBER OF RUNS PER DUMP
num_of_runs = 10

# AMOUNT OF TIMES THE RANDOM FILES LIST IS REPEATED TO INCREASE AMOUNT OF REQUESTS
repeat_factor = 10

# SPECIFIES THE CURRENT DUMP THAT IS BEING TESTED (DO NOT CHANGE)
dump_to_test = ""

# FETCH SIZES TO BE TESTED IN FETCH SIZES EXPERIMENT
fetch_sizes = [1, 10, 100, 1000, 5000, 10000]

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


# Grabs num_files random files from the dump_to_test
def get_random_files():
    all_files = []

    # Walk through the directory and its subfolders
    for root, _, files in os.walk(dumps_folder_dict.get(dump_to_test)):
        for file in files:
            # Add the file to the all_files list
            all_files.append(os.path.join(root, file))

    # If there are fewer files than num_files, return all of them
    if len(all_files) <= num_files:
        return all_files

    # Randomly select the specified number of files
    return random.sample(all_files, num_files)


# Extends the random files list, shuffles the list,
# creates a list with the appropriate data-structure for the experiment
def setup():
    result = dict()

    try:
        # Repeat list to increase # of fetches
        files_to_fetch = get_random_files() * repeat_factor
        # Shuffle the list for randomness
        random.shuffle(files_to_fetch)

        for protocol in ["http", "https"]:
            # Prepare a list of (protocol, file) tuples
            files_to_fetch = [(protocol, file) for file in files_to_fetch]
            # Store the files to be fetched in the result dictionary
            result[f"files_{protocol}"] = files_to_fetch

    except Exception as err:
        print("Error occurred:", err)
    except KeyboardInterrupt:
        print("Experiment interrupted by user...")

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
            with ThreadPoolExecutor(max_workers=threads) as executor:
                # Fetch each file using the specified protocol
                executor.map(lambda file: fetch(protocol, file), files)

            # Stop the tracker after the fetches are complete
            tracker.stop()

        except Exception as err:
            print("Error occurred:", err)
        except KeyboardInterrupt:
            print("Experiment interrupted by user...")


# Performs the file size experiment
def file_size_experiment():
    global dump_to_test
    dump_to_test = "wikipedia"
    all_files = []

    # Walk through the directory and its subfolders
    for root, _, files in os.walk(dumps_folder_dict.get(dump_to_test)):
        for file in files:
            # Add the file to the all_files list
            file_path = os.path.join(root, file)
            # Include file size as a tuple (path, size)
            all_files.append((file_path, os.path.getsize(file_path)))

    # Create a DataFrame
    files_df = pd.DataFrame(all_files, columns=["file_path", "file_size"])

    # Set the number of bins
    num_bins = 5
    # Number of files to select from each bin
    num_files_per_bin = 10

    # Create bins based on file sizes
    files_df["size_bin"] = pd.cut(files_df["file_size"], bins=num_bins)

    # Randomly select files from each bin
    selected_files = []
    for _, bin_group in files_df.groupby("size_bin"):
        # Take at most `num_files_per_bin` files from each bin
        selected_files.extend(
            bin_group.sample(n=min(len(bin_group), num_files_per_bin))[
                "file_path"
            ].tolist()
        )

    for protocol in ["http", "https"]:

        for iteration in range(num_of_runs):
            # Create a CodeCarbon offline tracker
            tracker = OfflineEmissionsTracker(
                measure_power_secs=1,
                country_iso_code="NLD",
                output_file=f"{project_path}/file_size_data_{iteration}_{protocol}.csv",
                log_level="error",
            )

            # Open the relevant results file
            with open(
                f"{project_path}/file_size_data_sizes_{iteration}_{protocol}.csv",
                "w",
                newline="",
            ) as output_file:
                writer = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["size"])

                for file in selected_files:
                    # Start the tracker before fetching
                    tracker.start()

                    # Fetch each file using the specified protocol
                    fetch(protocol, file)

                    # Stop the tracker after the fetches are complete
                    tracker.stop()

                    # Write size
                    writer.writerow([os.path.getsize(file)])


# Performs the fetch size experiment
def fetch_sizes_experiment():
    global dump_to_test
    dump_to_test = "wikipedia"
    selected_file = f"{dumps_folder_dict.get(dump_to_test)}/index.html"

    for protocol in ["http", "https"]:

        for iteration in range(num_of_runs):
            # Create a CodeCarbon offline tracker
            tracker = OfflineEmissionsTracker(
                measure_power_secs=1,
                country_iso_code="NLD",
                output_file=f"{project_path}/fetch_sizes_data_{iteration}_{protocol}.csv",
                log_level="error",
            )

            for i in range(len(fetch_sizes)):
                # Start the tracker before fetching
                tracker.start()

                for j in range(fetch_sizes[i]):
                    # Fetch the file using the specified protocol
                    fetch(protocol, selected_file[0])

                # Stop the tracker after the fetches are complete
                tracker.stop()


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
    for dump in dumps:
        # Store the absolute file path
        file_path = f"{project_path}/results_{dump}.csv"

        # Check if the file exists
        if os.path.exists(file_path):
            # Add the file to the files list
            files.append(file_path)

    # Grab the quantities to plot
    if len(files) > 0:
        df = pd.read_csv(files[0])
    else:
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
        http_variances = []
        https_variances = []

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
                    http_variance = sum(
                        (x - http_mean) ** 2 for x in http_values
                    ) / len(http_values)
                    https_variance = sum(
                        (x - https_mean) ** 2 for x in https_values
                    ) / len(https_values)

                    # Use the file name (without extension) as the label
                    test_name = (
                        file.replace(f"{project_path}/results_", "")
                        .replace(".csv", "")
                        .capitalize()
                    )

                    # Add the relevant values to the lists
                    x.append(test_name)
                    y_http.append(http_mean)
                    y_https.append(https_mean)
                    http_variances.append(http_variance)
                    https_variances.append(https_variance)

        # Compute standard deviations from variances
        http_std_devs = [math.sqrt(var) for var in http_variances]
        https_std_devs = [math.sqrt(var) for var in https_variances]

        # Plot grouped bar chart for the current quantity
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar width and index adjustment
        bar_width = 0.35  # Width of each bar
        index = range(len(x))

        # Create bars for HTTP and HTTPS (with error bars for variances)
        ax.bar(index, y_http, bar_width, label=f"HTTP {quantity}", yerr=http_std_devs)
        ax.bar(
            [i + bar_width for i in index],
            y_https,
            bar_width,
            label=f"HTTPS {quantity}",
            yerr=https_std_devs,
        )

        # Add labels, title, and legend with increased font sizes
        ax.set_xlabel("Websites")
        ax.set_ylabel(f"{quantity_name} ({quantity_unit})")
        ax.set_xticks(
            [i + bar_width / 2 for i in index]
        )  # Adjust x-ticks to be between bars
        ax.set_xticklabels(
            x, rotation=45, ha="right"
        )  # Set x-axis labels with rotation for readability

        # Create a list containing all HTTP and HTTPS values (alternating between HTTP and HTTPS)
        all_values = list(itertools.chain(*zip(y_http, y_https)))

        # Annotate each bar with increased font size
        for i, value in enumerate(all_values):
            # Determine the x position of the text
            # HTTP value
            if i % 2 == 0:
                x_pos = i // 2

            # HTTPS value
            else:
                x_pos = i // 2 + bar_width

            # Check if the value is smaller than 1e-6
            if abs(value) < 1e-3:
                # Use scientific notation
                display_value = f"{value:.2e}"
            else:
                # Use regular floating-point format
                display_value = f"{value:.2f}"

            # Put the annotation in the correct place
            plt.text(x_pos, 1.01 * value, display_value, ha="center")

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


# Generates a scatter plot showcasing file size vs emissions
def generate_file_size_plot():
    # Initialize lists to store emissions and file sizes for each run
    http_emissions_runs = []
    https_emissions_runs = []
    http_sizes_runs = []
    https_sizes_runs = []

    # Load all iterations for HTTP and HTTPS
    for i in range(num_of_runs):
        # Read emissions and file sizes for each run
        http_emissions = pd.read_csv(f"{project_path}/file_size_data_{i}_http.csv")[
            "emissions"
        ]
        https_emissions = pd.read_csv(f"{project_path}/file_size_data_{i}_https.csv")[
            "emissions"
        ]
        http_sizes = pd.read_csv(f"{project_path}/file_size_data_sizes_{i}_http.csv")[
            "size"
        ]
        https_sizes = pd.read_csv(f"{project_path}/file_size_data_sizes_{i}_https.csv")[
            "size"
        ]

        # Append data for this run
        http_emissions_runs.append(http_emissions)
        https_emissions_runs.append(https_emissions)
        http_sizes_runs.append(http_sizes)
        https_sizes_runs.append(https_sizes)

    # Convert the list of emissions and sizes into DataFrames
    http_emissions_df = pd.DataFrame(http_emissions_runs)
    https_emissions_df = pd.DataFrame(https_emissions_runs)
    http_sizes_df = pd.DataFrame(http_sizes_runs)
    https_sizes_df = pd.DataFrame(https_sizes_runs)

    # Compute the average emissions and file sizes across all runs
    http_emissions_mean = http_emissions_df.mean(axis=0)
    https_emissions_mean = https_emissions_df.mean(axis=0)
    http_sizes_mean = http_sizes_df.mean(axis=0)
    https_sizes_mean = https_sizes_df.mean(axis=0)

    # Create scatter plot
    plt.figure(figsize=(10, 6))

    # Plot HTTP and HTTPS data
    plt.scatter(http_sizes_mean, http_emissions_mean, label="HTTP", alpha=0.7)
    plt.scatter(https_sizes_mean, https_emissions_mean, label="HTTPS", alpha=0.7)

    # Draw lines connecting HTTP and HTTPS for each file
    for i in range(len(http_sizes_mean)):
        plt.plot(
            [http_sizes_mean[i], https_sizes_mean[i]],
            [http_emissions_mean[i], https_emissions_mean[i]],
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )

    # Labeling with increased font sizes
    plt.title("Scatter Plot of File Size vs Emissions")
    plt.xlabel("File Size (bytes)")
    plt.ylabel("Emissions (kgCO₂eq)")

    # Adjust legend with larger font size
    plt.legend()

    # Adjust tick labels font size
    plt.xticks()
    plt.yticks()

    plt.tight_layout()

    # Delete the plot if it already exists
    plot_path = f"{project_path}/file_size_plot.png"
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Save the new plot
    plt.savefig(plot_path, format="png", bbox_inches="tight")


# Generates a plot showcasing fetch sizes vs emissions
def generate_fetch_sizes_plot():
    # Initialize lists to store emissions data
    http_emissions_runs = []
    https_emissions_runs = []

    # Load all iterations for HTTP and HTTPS
    for i in range(num_of_runs):
        # Read HTTP and HTTPS data for each run
        http_data = pd.read_csv(f"{project_path}/fetch_sizes_data_{i}_http.csv")
        https_data = pd.read_csv(f"{project_path}/fetch_sizes_data_{i}_https.csv")

        # Append emissions data for this run
        http_emissions_runs.append(http_data["emissions"])
        https_emissions_runs.append(https_data["emissions"])

    # Convert the list of emissions runs into DataFrames
    http_emissions_df = pd.DataFrame(http_emissions_runs)
    https_emissions_df = pd.DataFrame(https_emissions_runs)

    # Compute the average emissions across all runs for each fetch size
    http_emissions_mean = http_emissions_df.mean(axis=0)
    https_emissions_mean = https_emissions_df.mean(axis=0)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(
        fetch_sizes,
        http_emissions_mean,
        label="HTTP",
    )
    plt.plot(
        fetch_sizes,
        https_emissions_mean,
        label="HTTPS",
    )

    # Add labels, title, and legend
    plt.title("Plot of Fetch Size vs Emissions")
    plt.xlabel("Fetch Size (files)")
    plt.ylabel("Emissions (kgCO₂eq)")
    plt.legend()

    # Increase tick label font sizes
    plt.xticks()
    plt.yticks()

    plt.tight_layout()

    # Delete the plot if it already exists
    plot_path = f"{project_path}/fetch_sizes_plot.png"
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
            f"{os.linesep}{os.linesep}Starting {num_of_runs} runs of the {dump_to_test} dump..."
        )
        # Perform num_of_runs runs
        for i in range(num_of_runs):
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

    # Run the file size experiment
    print(f"{os.linesep}Starting the file size experiment...")
    file_size_experiment()

    # Wait until every file_size_data file is generated
    print(
        f"{os.linesep}{os.linesep}File size experiment finished! Waiting for all results files to be generated..."
    )
    all_file_size_files = [f"file_size_data_{i}_http" for i in range(num_of_runs)]
    all_file_size_files.extend(
        [f"file_size_data_{i}_https" for i in range(num_of_runs)]
    )
    all_file_size_files.extend(
        [f"file_size_data_sizes_{i}_http" for i in range(num_of_runs)]
    )
    all_file_size_files.extend(
        [f"file_size_data_sizes_{i}_https" for i in range(num_of_runs)]
    )
    for file in all_file_size_files:
        while True:
            if Path(f"{project_path}/{file}.csv").exists():
                break
            else:
                sleep(0.1)

    # Generate the scatter plot showcasing file size versus emissions
    print(
        f"{os.linesep}Generating scatter plot showcasing file size versus emissions..."
    )
    generate_file_size_plot()

    # Run the fetch sizes experiment
    print(f"{os.linesep}Starting the fetch sizes experiment...")
    fetch_sizes_experiment()

    # Wait until every fetch_sizes_data file is generated
    print(
        f"{os.linesep}{os.linesep}Fetch sizes experiment finished! Waiting for all results files to be generated..."
    )
    all_fetch_sizes_files = [f"fetch_sizes_data_{i}_http" for i in range(num_of_runs)]
    all_fetch_sizes_files.extend(
        [f"fetch_sizes_data_{i}_https" for i in range(num_of_runs)]
    )
    for file in all_fetch_sizes_files:
        while True:
            if Path(f"{project_path}/{file}.csv").exists():
                break
            else:
                sleep(0.1)

    # Generate the plot showcasing fetch sizes versus emissions
    print(f"{os.linesep}Generating plot showcasing fetch sizes versus emissions...")
    generate_fetch_sizes_plot()

    # End of the experiment script
    print(
        f"{os.linesep}All experiments finished!{os.linesep}Terminating...{os.linesep}"
    )

# ----------------------------------------------------END OF SCRIPT-----------------------------------------------------#

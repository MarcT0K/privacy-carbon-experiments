import itertools
import random
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import requests
from time import sleep
from pathlib import Path
from sys import exit
from codecarbon import OfflineEmissionsTracker
from concurrent.futures import ThreadPoolExecutor

#TERMINAL EXECUTION COMMAND
#sudo /path/to/your/virtualenv/bin/python path/to/your/project_folder/experiment.py

#--------------------------------------------------START OF VARIABLES--------------------------------------------------#

#PROJECT_PATH
project_path = os.path.dirname(__file__)

#CERTIFICATE PATH
cert_path = "/etc/nginx/ssl/localhost.crt"

#USER DIRECTORY PATH
user = os.environ.get("SUDO_USER")
home_dir = os.path.expanduser(f"~{user}")

#SERVER URL
base_url = "localhost"

#AMOUNT OF THREADS
threads = 8

#LIST OF ALL DUMPS
dumps = ["wikipedia", "nytimes", "github", "youtube", "amazon"]

#RELEVANT FOLDER TO TEST FOR EACH DUMP
dumps_folder_dict = dict()
dumps_folder_dict["wikipedia"] = f"{home_dir}/Downloads/wikipedia-simple-html/simple/"
dumps_folder_dict["nytimes"] = f"{home_dir}/Downloads/nytimes/www.nytimes.com"
dumps_folder_dict["github"] = f"{home_dir}/Downloads/github/github.com"
dumps_folder_dict["youtube"] = f"{home_dir}/Downloads/youtube/www.youtube.com"
dumps_folder_dict["amazon"] = f"{home_dir}/Downloads/amazon/www.amazon.nl"

#NUMBER OF RANDOM FILES USED FROM DUMP
num_files = 20

#NUMBER OF RUNS PER DUMP
num_of_runs = 10

#AMOUNT OF TIMES FILES LIST IS REPEATED TO INCREASE # OF REQUESTS
repeat_factor = 1000

#PERFORM 1 RUN OF dump_to_test, OR PERFORM THE ENTIRE EXPERIMENT (10 runs of every dump)
single_run = True

#SPECIFY THE DUMP TO RUN THE EXPERIMENT ON [wikipedia/nytimes/github/youtube/amazon]
#(Only applicable if single_run = True)
dump_to_test = "wikipedia"

#---------------------------------------------------END OF VARIABLES---------------------------------------------------#

#--------------------------------------------------START OF FUNCTIONS--------------------------------------------------#

def get_random_files():
    all_files = []
    print(dumps_folder_dict.get(dump_to_test))

    # Walk through the directory and its subfolders
    for root, _, files in os.walk(dumps_folder_dict.get(dump_to_test)):
        for file in files:
            all_files.append(os.path.join(root, file))

    # If there are fewer files than requested, return all of them
    if len(all_files) <= num_files:
        return all_files

    # Randomly select the specified number of files
    return random.sample(all_files, num_files)


def fetch(protocol, file):
    fetch_url = f"{protocol}://{base_url}/{file}"
    if protocol == "http":
        response = requests.get(fetch_url)
    else:
        response = requests.get(fetch_url, verify=cert_path)
    return response


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
            result[f"files_{protocol}"] = files_to_fetch

    except Exception as err:
        print("Error occurred:", err)
    except KeyboardInterrupt:
        print("Experiment interrupted by user...")

    return result


def experiment():
    setup_dict = setup()

    tracker = OfflineEmissionsTracker(
        measure_power_secs=1,
        country_iso_code="NLD",
        output_file=f"{project_path}/raw_emissions_{dump_to_test}.csv",
        log_level="error"
    )

    for protocol in ["http", "https"]:
        try:
            # Store the tracker
            # tracker = dict.get(f"tracker_{protocol}")

            #Store the files to be fetched
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


def generate_plots():
    # Define the files and the quantities to plot
    files = [
        "results_wikipedia.csv",
        "results_amazon.csv",
        "results_nytimes.csv",
        "results_github.csv",
        "results_youtube.csv"
    ]

    quantities = [
        "duration",
        "emissions",
        "emissions_rate",
        "energy_consumed",
        "cpu_power",
        "cpu_energy",
        "ram_power",
        "ram_energy"
    ]

    quantities_units = dict()
    quantities_units["duration"] = "s"
    quantities_units["emissions"] = "kgCO₂eq"
    quantities_units["emissions_rate"] = "kgCO₂eq/s"
    quantities_units["energy_consumed"] = "kWh"
    quantities_units["cpu_power"] = "W"
    quantities_units["cpu_energy"] = "kWh"
    quantities_units["ram_power"] = "W"
    quantities_units["ram_energy"] = "kWh"

    # Iterate over each quantity and generate the plots
    for quantity in quantities:
        x = []  # Labels for each pair of bars
        y_http = []  # Values for HTTP
        y_https = []  # Values for HTTPS

        # Extract data for each quantity
        for file in files:
            file_path = f"{project_path}/{file}"

            if os.path.exists(file_path):

                with open(file_path, 'r') as existingFile:
                    reader = csv.reader(existingFile, quoting=csv.QUOTE_MINIMAL)
                    next(reader, None)  # Skip header

                    # Initialize variables for the quantity data
                    http_value = None
                    https_value = None

                    # Find HTTP and HTTPS values for the current quantity
                    for row in reader:
                        if row[0] == 'HTTP':  # Assuming first column has protocol names
                            http_value = float(
                                row[quantities.index(quantity) + 1])  # Adjust column index based on quantity
                        elif row[0] == 'HTTPS':
                            https_value = float(row[quantities.index(quantity) + 1])

                    # Only append if both HTTP and HTTPS data are found for the file
                    if http_value is not None and https_value is not None:
                        # Use the file name (without extension) as the label
                        test_name = file.replace('results_', '').replace('.csv', '').capitalize()
                        x.append(test_name)
                        y_http.append(http_value)
                        y_https.append(https_value)

        # Plot grouped bar chart for the current quantity
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar width and index adjustment
        bar_width = 0.35  # Width of each bar
        index = range(len(x))

        # Create bars for HTTP and HTTPS (next to each other)
        ax.bar(index, y_http, bar_width, color='b', label=f'HTTP {quantity}')
        ax.bar([i + bar_width for i in index], y_https, bar_width, color='g', label=f'HTTPS {quantity}')

        # Add labels, title, and legend
        ax.set_xlabel('Tests')
        ax.set_ylabel(f'{quantity.capitalize()} ({quantities_units[quantity]})')
        ax.set_title(f'{quantity.capitalize()} of HTTP vs HTTPS')
        ax.set_xticks([i + bar_width / 2 for i in index])  # Adjust x-ticks to be between bars
        ax.set_xticklabels(x, rotation=45, ha='right')  # Set x-axis labels with rotation for readability

        all_values = list(itertools.chain(*zip(y_http, y_https)))
        # Annotate each bar
        for i, value in enumerate(all_values):
            # Determine the x position of the text
            if i % 2 == 0:  # HTTP value
                x_pos = i // 2
            else:  # HTTPS value
                x_pos = i // 2 + bar_width

            plt.text(x_pos, 1.01 * value, round(value, 6), ha='center', fontsize=6)

        # Move the legend outside the plot area to avoid overlap with the bars
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)

        # Save the plot
        plot_path = f"{project_path}/{quantity}_plot.png"
        if os.path.exists(plot_path):
            os.remove(plot_path)

        plt.savefig(plot_path, bbox_inches='tight')


def gather_results():
    # Fetch the results
    df = pd.read_csv(f'{project_path}/raw_emissions_{dump_to_test}.csv')

    # Grab header row and separate http/https results
    header = df.columns
    df_http = pd.DataFrame(df.iloc[::2].values, columns=header)
    df_https = pd.DataFrame(df.iloc[1::2].values, columns=header)

    # Generate the output file
    with open (f'{project_path}/results_{dump_to_test}.csv', 'w', newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['', 'duration', 'emissions', 'emissions_rate', 'energy_consumed',
                         'cpu_power', 'cpu_energy', 'ram_power', 'ram_energy'])
        writer.writerow(['HTTP', df_http['duration'].mean(), df_http['emissions'].mean(),
                         df_http['emissions_rate'].mean(), df_http['energy_consumed'].mean(),
                         df_http['cpu_power'].mean(), df_http['cpu_energy'].mean(),
                         df_http['ram_power'].mean(), df_http['ram_energy'].mean()])
        writer.writerow(['HTTPS', df_https['duration'].mean(), df_https['emissions'].mean(),
                         df_https['emissions_rate'].mean(), df_https['energy_consumed'].mean(),
                         df_https['cpu_power'].mean(), df_https['cpu_energy'].mean(),
                         df_https['ram_power'].mean(), df_https['ram_energy'].mean()])


#---------------------------------------------------END OF FUNCTIONS---------------------------------------------------#

#WARNING: THIS EXPERIMENT REQUIRES ROOT (SUDO)
if __name__ == "__main__":
    # Check if running as admin
    if os.geteuid() != 0:
        print("This script needs root in order to generate valid results!")
        exit(1)

    # Perform a single run
    if single_run:
        # Run the experiment once
        experiment()

        # Wait until raw_emissions_{dump}.csv is generated (if needed)
        while not Path(f"{project_path}/raw_emissions_{dump_to_test}.csv").exists():
            sleep(0.1)

        # Put the relevant results into the results_{dump_to_test}.csv file
        gather_results()

    # Run entire experiment
    elif not single_run:
        # Iterate through every dump
        for dump in dumps:
            # Set the dump_to_test to the dump to test
            dump_to_test = dump

            # Perform num_of_runs runs
            for i in range(num_of_runs):
                experiment()

            # Wait until raw_emissions_{dump}.csv is generated (if needed)
            while not Path(f"{project_path}/raw_emissions_{dump_to_test}.csv").exists():
                sleep(0.1)
            
            # Put the relevant results into the results_{dump_to_test}.csv file
            gather_results()

        # Generate plots of every column of relevant results for every dump
        generate_plots()



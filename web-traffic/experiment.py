import itertools
import random
import warnings
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

#UNITS OF EVERY RELEVANT QUANTITY IN RESULTS FILE
quantities_units = {"duration": "s", "emissions": "kgCO₂eq",
                    "emissions_rate": "kgCO₂eq/s", "energy_consumed": "kWh",
                    "cpu_power": "W", "cpu_energy": "kWh", "ram_power": "W",
                    "ram_energy": "kWh"}

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

#AMOUNT OF TIMES THE RANDOM FILES LIST IS REPEATED TO INCREASE AMOUNT OF REQUESTS
repeat_factor = 1000

#PERFORM SINGLE RUN OF dump_to_test IF TRUE, PERFORM ENTIRE EXPERIMENT IF FALSE
single_run = False

#SPECIFY THE DUMP TO RUN THE EXPERIMENT ON (check dumps list for all valid dumps)
#(Only applicable if single_run = True)
dump_to_test = "wikipedia"

#---------------------------------------------------END OF VARIABLES---------------------------------------------------#

#--------------------------------------------------START OF FUNCTIONS--------------------------------------------------#

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


def experiment():
    # Suppress warnings from pandas used in CodeCarbon as they are irrelevant
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Call the setup function
    setup_dict = setup()

    # Create a CodeCarbon offline tracker
    tracker = OfflineEmissionsTracker(
        measure_power_secs=1,
        country_iso_code="NLD",
        output_file=f"{project_path}/raw_emissions_{dump_to_test}.csv",
        log_level="error"
    )

    # Run the experiment for both HTTP and HTTPS
    for protocol in ["http", "https"]:
        try:
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
        quantities = list(df.columns[1:])
    else:
        print("No results files available. First create results before generating plots...")
        return

    # Iterate over each quantity and generate the plots
    for quantity in quantities:
        # Create lists for the labels, HTTP values and HTTPS values
        x = []
        y_http = []
        y_https = []

        # Extract data for each quantity
        for file in files:
            with open(file, 'r') as existingFile:
                reader = csv.reader(existingFile, quoting=csv.QUOTE_MINIMAL)
                # Skip the header
                next(reader, None)

                # Initialize variables for the quantity data
                http_value = None
                https_value = None

                # Find HTTP and HTTPS values for the current quantity
                for row in reader:
                    if row[0] == 'HTTP':
                        # (Adjust column index based on quantity)
                        http_value = float(
                            row[quantities.index(quantity) + 1])
                    elif row[0] == 'HTTPS':
                        # (Adjust column index based on quantity)
                        https_value = float(row[quantities.index(quantity) + 1])

                # Only append if both HTTP and HTTPS data are found for the file
                if http_value is not None and https_value is not None:
                    # Use the file name (without extension) as the label
                    test_name = file.replace(f"{project_path}/results_", "").replace('.csv', '').capitalize()
                    # Add the relevant values to the lists
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

        # Create a list containing all HTTP and HTTPS values (alternating between HTTP and HTTPS)
        all_values = list(itertools.chain(*zip(y_http, y_https)))

        # Annotate each bar
        for i, value in enumerate(all_values):
            # Determine the x position of the text
            # HTTP value
            if i % 2 == 0:
                x_pos = i // 2

            # HTTPS value
            else:
                x_pos = i // 2 + bar_width

            # Put the annotation in the correct place
            plt.text(x_pos, 1.01 * value, round(value, 6), ha='center', fontsize=6)

        # Move the legend outside the plot area to avoid overlap with the bars
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)

        # Delete the plot if it already exists
        plot_path = f"{project_path}/{quantity}_plot.png"
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # Save the new plot
        plt.savefig(plot_path, bbox_inches='tight')


def gather_results():
    # Fetch the results
    df = pd.read_csv(f'{project_path}/raw_emissions_{dump_to_test}.csv')

    # Grab header row and separate http/https results
    header = df.columns
    df_http = pd.DataFrame(df.iloc[::2].values, columns=header)
    df_https = pd.DataFrame(df.iloc[1::2].values, columns=header)

    # Open the relevant results file
    with open (f'{project_path}/results_{dump_to_test}.csv', 'w', newline="") as file:
        # Create a writer for the file
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # Write the header row
        writer.writerow(['', 'duration', 'emissions', 'emissions_rate', 'energy_consumed',
                         'cpu_power', 'cpu_energy', 'ram_power', 'ram_energy'])
        # Write the HTTP values row
        writer.writerow(['HTTP', df_http['duration'].mean(), df_http['emissions'].mean(),
                         df_http['emissions_rate'].mean(), df_http['energy_consumed'].mean(),
                         df_http['cpu_power'].mean(), df_http['cpu_energy'].mean(),
                         df_http['ram_power'].mean(), df_http['ram_energy'].mean()])
        # Write the HTTPS values row
        writer.writerow(['HTTPS', df_https['duration'].mean(), df_https['emissions'].mean(),
                         df_https['emissions_rate'].mean(), df_https['energy_consumed'].mean(),
                         df_https['cpu_power'].mean(), df_https['cpu_energy'].mean(),
                         df_https['ram_power'].mean(), df_https['ram_energy'].mean()])

    # Open the relative ratio (HTTPS vs HTTP) file
    with open (f'{project_path}/ratios_{dump_to_test}.csv', 'w', newline="") as file:
        # Create a writer for the file
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # Write the header row
        writer.writerow(['', 'duration', 'emissions', 'emissions_rate', 'energy_consumed',
                         'cpu_power', 'cpu_energy', 'ram_power', 'ram_energy'])
        # Write the relative ratios row
        writer.writerow(['Relative Ratio (HTTPS vs HTTP)', df_https['duration'].mean() / df_http['duration'].mean(),
                         df_https['emissions'].mean() / df_http['emissions'].mean(),
                         df_https['emissions_rate'].mean() / df_http['emissions_rate'].mean(),
                         df_https['energy_consumed'].mean() / df_http['energy_consumed'].mean(),
                         df_https['cpu_power'].mean() / df_http['cpu_power'].mean(),
                         df_https['cpu_energy'].mean() / df_http['cpu_energy'].mean(),
                         df_https['ram_power'].mean() / df_http['ram_power'].mean(),
                         df_https['ram_energy'].mean() / df_http['ram_energy'].mean()
                         ])


#---------------------------------------------------END OF FUNCTIONS---------------------------------------------------#

#WARNING: THIS EXPERIMENT REQUIRES ROOT (SUDO)
if __name__ == "__main__":
    # Check if running as admin
    if os.geteuid() != 0:
        # Experiment is started without root
        print("This script needs root in order to generate valid results!")
        # Abort the experiment
        exit(1)

    # Perform a single run
    if single_run:
        # Run the experiment once
        print(f"{dump_to_test}Starting single run of the {dump_to_test} dump...")
        experiment()
        print(f"{os.linesep}Run finished!")

        # Wait until raw_emissions_{dump}.csv is generated (if needed)
        print(f"{os.linesep}Waiting for {project_path}/raw_emissions_{dump_to_test}.csv to be generated...")
        while not Path(f"{project_path}/raw_emissions_{dump_to_test}.csv").exists():
            sleep(0.1)

        # Put the relevant results into the results_{dump_to_test}.csv file
        print(f"{os.linesep}Writing relevant results to results_{dump_to_test}.csv...")
        gather_results()

        # End of single_run
        print(f"{os.linesep}Single run finished!{os.linesep}")

    # Run entire experiment
    elif not single_run:
        print(f"{os.linesep}Starting full experiment...")
        # Iterate through every dump
        for dump in dumps:
            # Set the dump_to_test to the dump to test
            dump_to_test = dump

            print(f"{os.linesep}{os.linesep}Starting {num_of_runs} runs of the {dump_to_test} dump...")
            # Perform num_of_runs runs
            for i in range(num_of_runs):
                experiment()
                print(f"Run {i} finished!")

            # Wait until raw_emissions_{dump}.csv is generated (if needed)
            print(f"{os.linesep}Waiting for raw_emissions_{dump_to_test}.csv to be generated...")
            while not Path(f"{project_path}/raw_emissions_{dump_to_test}.csv").exists():
                sleep(0.1)

            # Put the relevant results into the results_{dump_to_test}.csv file
            print(f"{os.linesep}Writing relevant results to results_{dump_to_test}.csv...")
            gather_results()

        # Wait until every results file is generated
        print(f"{os.linesep}{os.linesep}All runs finished! Waiting for all results files to be generated...")
        for dump in dumps:
            while True:
                if Path(f"{project_path}/results_{dump}.csv").exists():
                    break
                else:
                    sleep(0.1)

        # Generate plots of every column of relevant results for every dump
        print(f"{os.linesep}Generating combined bar plots for every individual variable in results files...")
        generate_plots()

        # End of full experiment
        print(f"{os.linesep}Full experiment finished!{os.linesep}")


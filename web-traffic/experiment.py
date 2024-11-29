import random
import pandas as pd
import csv
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

#SERVER URL
base_url = "localhost"

#AMOUNT OF THREADS
threads = 4

#AMOUNT OF TIMES FILES LIST IS REPEATED TO INCREASE # OF REQUESTS
repeat_factor = 1000

#LIST OF FILES TO TEST FOR WIKIPEDIA
wikipedia_files = [
    "index.html", #main HTML page
    "COPYING.html", #licensing HTML page
    "c/o/m/Wikipedia~CommonsTicker_2086.html", #biggest file of the dump
    "t/b/c/User~TBC_Lupin's_popups.js_af18.html", #second-biggest file of the dump
    "a/d/_/AD_1_8da8.html", #smallest html file
    "n/e/t/Netherlands.html", #information-containing HTML page
    "e/n/g/England.html" #Information-containing HTML page
    "skins/common/sticky.js", #JS file
    "skins/common/protect.js", #JS file
    "skins/monobook/main.css", #CSS file
    "skins/common/common.css", #CSS file
    "skins/disabled/HTMLDump.php.broken", #PHP file
    "skins/Skin.sample", #PHP file
    "skins/drsport/wp_logo.gif", #GIF file
    "skins/monobook/file_icon.gif", #GIF file
    "skins/monobook/headbg.jpg", #JPG file
    "skins/drsport/graphics/bg_body.jpg", #JPG file
    "images/wiki-en.png", #PNG file
    "skins/common/images/button_nowiki.png" #PNG file
    "n", #Folder
    "skins" #Folder
]

#DICTIONARY FOR FILES PER DUMP
files_dict = dict()
files_dict["wikipedia"] = wikipedia_files
# files_dict[""] =

# SPECIFY THE DUMP TO RUN THE EXPERIMENT ON [wikipedia/...]
dump_to_test = "wikipedia"

#---------------------------------------------------END OF VARIABLES---------------------------------------------------#

#--------------------------------------------------START OF FUNCTIONS--------------------------------------------------#

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
        files_to_fetch = files_dict.get(dump_to_test) * repeat_factor
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


def gather_results():
    # Fetch the results
    df = pd.read_csv(f'{project_path}/raw_emissions_{dump_to_test}.csv')

    # Grab header row and separate http/https results
    header = df.columns
    df_http = pd.DataFrame(df.iloc[::2].values, columns=header)
    df_https = pd.DataFrame(df.iloc[1::2].values, columns=header)

    # Generate the output file
    with open (f'{project_path}/results_{dump_to_test}.csv', 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['', 'duration', 'emissions', 'emissions_rate', 'cpu_power'])
        writer.writerow(['HTTP', df_http['duration'].mean(), df_http['emissions'].mean(),
                         df_http['emissions_rate'].mean(), df_http['cpu_power'].mean()])
        writer.writerow(['HTTPS', df_https['duration'].mean(), df_https['emissions'].mean(),
                         df_https['emissions_rate'].mean(), df_https['cpu_power'].mean()])

#---------------------------------------------------END OF FUNCTIONS---------------------------------------------------#

#WARNING: THIS EXPERIMENT REQUIRES ROOT (SUDO)
if __name__ == "__main__":
    # Check if running as admin
    if not os.environ.get('USERPROFILE'):
        print("This script needs to be run as an administrator!")
        exit(1)

    # Run the experiment
    experiment()

    # Wait until raw_emissions_{dump}.csv is generated (if needed)
    while not Path(f"{project_path}/raw_emissions_{dump_to_test}.csv").exists():
        sleep(0.1)

    # Gather the results from the csv files
    gather_results()
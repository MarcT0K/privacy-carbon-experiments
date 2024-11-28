import random
import pandas as pd
import csv
import os
import requests
from time import sleep
from codecarbon import OfflineEmissionsTracker
from concurrent.futures import ThreadPoolExecutor

#TERMINAL RUN COMMAND (USE SUDO!)
#sudo ~/University/PycharmProjects/Research_Project/.venv/bin/python ~/University/PycharmProjects/Research_Project/experiment.py

#PROJECT_PATH
project_path = os.path.dirname(__file__)

#CERTIFICATE PATH
cert_path = "/etc/nginx/ssl/localhost.crt"

#SERVER URL
base_url = "localhost"

#AMOUNT OF THREADS
threads = 4

#AMOUNT OF TIMES FILES LIST IS REPEATED
repeat_factor = 1000

#LIST OF FILES TO TEST FOR WIKIMEDIA
wikimedia_files = [
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
    # "n", #Folder
    # "skins" #Folder
]

#DICTIONARY FOR FILES PER DUMP
files_dict = dict()
files_dict["wikimedia"] = wikimedia_files
# files_dict[""] =


def fetch(protocol, file):
    fetch_url = f"{protocol}://{base_url}/{file}"
    if protocol == "http":
        response = requests.get(fetch_url)
    else:
        response = requests.get(fetch_url, verify=cert_path)
    return response


def read_energy(file_path):
    """Reads the energy value from the specified file."""
    try:
        with open(file_path, "r") as file:
            return int(file.read().strip())
    except Exception as e:
        print(f"Error reading energy file: {e}")
        return None


def experiment(dump):
    try:
        # Repeat list to increase # of fetches
        files_to_fetch = files_dict.get(dump) * repeat_factor
        # Shuffle the list for randomness
        random.shuffle(files_to_fetch)

        for protocol in ["http", "https"]:
            # Create tracker based on protocol
            tracker = OfflineEmissionsTracker(
                measure_power_secs=0.1,
                country_iso_code="NLD",
                output_file=f"{project_path}/raw_emissions_{dump}_{protocol}.csv",
                log_level="error"
            )

            # Prepare a list of (protocol, file) tuples
            files_to_fetch = [(protocol, file) for file in files_to_fetch]

            # Perform the fetches using a thread pool
            with ThreadPoolExecutor(max_workers=threads) as executor:
                # Start the tracker before fetching
                tracker.start()
                # Fetch each file using the specified protocol
                executor.map(lambda file: fetch(protocol, file), files_to_fetch)
                # Stop the tracker after the fetches are complete
                tracker.stop()

    except Exception as err:
        print("Error occurred:", err)
    except KeyboardInterrupt:
        print("Experiment interrupted by user...")


def gather_file_results(dump, protocol):
    df = pd.read_csv(f'{project_path}/raw_emissions_{dump}_{protocol}.csv')

    durations_average = df['duration'].mean() * 60  # mins to secs
    emissions_average = df['emissions'].mean()
    emissions_rate_average = df['emissions_rate'].mean()
    cpu_power_average = df['cpu_power'].mean()

    return [durations_average, emissions_average, emissions_rate_average, cpu_power_average]


def gather_results(dump):
    #HTTP------------------------------------------------------------------------
    http_results = gather_file_results(dump, "http")
    #HTTPS-----------------------------------------------------------------------
    https_results = gather_file_results(dump, "https")
    #OUTPUT_FILE------------------------------------------------------------------
    with open (f'{project_path}/results.csv', 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['', 'duration', 'emissions', 'emissions_rate', 'cpu_power'])
        writer.writerow(['HTTP', http_results[0], http_results[1], http_results[2], http_results[3]])
        writer.writerow(['HTTPS', https_results[0], https_results[1], https_results[2], https_results[3]])


if __name__ == "__main__":
    #SPECIFY THE DUMP TO RUN THE EXPERIMENT ON [wikimedia/...]
    dump_to_test = "wikimedia"

    #run the experiment
    experiment(dump_to_test)

    # sleep so that raw_emissions_http.csv and raw_emissions_https.csv can be generated
    sleep(3)

    #gather the results from the csv files
    gather_results(dump_to_test)
import random

import requests
from codecarbon import OfflineEmissionsTracker
from concurrent.futures import ThreadPoolExecutor

#TERMINAL RUN COMMAND
#~/University/PycharmProjects/Research_Project/.venv/bin/python ~/University/PycharmProjects/Research_Project/experiment.py

#CERTIFICATE PATH
cert_path = "/etc/nginx/ssl/localhost.crt"

#SERVER URL
base_url = "://localhost"

#AMOUNT OF THREADS
threads = 4

#AMOUNT OF # FILES LIST IS REPEATED
repeat_factor = 1000

#LIST OF FILES TO TEST
files = [
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
]

def fetch(protocol, file):
    fetch_url = f"{protocol}://{base_url}/{file}"
    if protocol == "http":
        response = requests.get(fetch_url)
    else:
        response = requests.get(fetch_url, verify=cert_path)
    return response


def experiment():
    try:
        # Repeat list to increase # of fetches
        files_to_fetch = files * repeat_factor
        # Shuffle the list for randomness
        random.shuffle(files_to_fetch)

        for protocol in ["http", "https"]:
            # Create tracker based on protocol
            tracker = OfflineEmissionsTracker(
                measure_power_secs=1,
                country_iso_code="NLD",
                output_file=f"/home/merijnposthuma/University/PycharmProjects/Research_Project/raw_emissions_{protocol}.csv",
                log_level="error"
            )

            # Prepare a list of (protocol, file) tuples
            files_to_fetch = [(protocol, file) for file in files_to_fetch]

            # Perform the fetches using a thread pool
            with ThreadPoolExecutor(max_workers=threads) as executor:
                #Start the tracker before fetching
                tracker.start()
                # Fetch each file using the specified protocol
                results = executor.map(lambda file: fetch(protocol, file), files_to_fetch)
                # Stop the tracker after the fetches are complete
                tracker.stop()


    except Exception as err:
        print("Error occurred:", err)
    except KeyboardInterrupt:
        print("Experiment stopped by user...")



if __name__ == "__main__":
    experiment()


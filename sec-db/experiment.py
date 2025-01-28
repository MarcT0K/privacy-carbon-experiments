import subprocess
from codecarbon import EmissionsTracker

# Function to run a script inside a pre-existing Docker container
def run_script_in_container():
    container_name = "bachelorresearch-redis-1"  # Replace with your container name
    script_path_plaintext = "/SWiSSSE/scripts/setup_plaintext.sh"  # Path to the script inside the container
    script_path_swissse = "/SWiSSSE/scripts/setup_swissse.sh"  # Path to the script inside the container

    # Execute the Python script inside the container
    subprocess.run([
        "docker", "exec", container_name, "sh", script_path_swissse
    ], check=True)

# Measure the environmental impact, plot results, and save to CSV
def measure_and_plot():
    tracker = EmissionsTracker()
    tracker.start()

    # Run the script inside the container
    run_script_in_container()

    emissions = tracker.stop()

if __name__ == "__main__":
    measure_and_plot()

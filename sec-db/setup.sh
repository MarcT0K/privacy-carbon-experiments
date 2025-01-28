#!/bin/bash

# Step 1: Clone the SWiSSSE repository
echo "Cloning the SWiSSSE repository..."
git clone --branch master https://github.com/SWiSSSE-crypto/SWiSSSE.git

# Navigate into the cloned repository
cd SWiSSSE || { echo "Failed to enter SWiSSSE directory"; exit 1; }

# Step 2: Download the Enron dataset
ENRON_ARCHIVE_URL="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
RAW_EMAILS_DIR="raw_emails"
echo "Downloading Enron email dataset..."
curl -O $ENRON_ARCHIVE_URL || { echo "Failed to download the archive"; exit 1; }

# Step 3: Extract the archive to /raw_emails/
echo "Extracting Enron email dataset..."
mkdir -p $RAW_EMAILS_DIR
tar -xvzf enron_mail_20150507.tar.gz -C $RAW_EMAILS_DIR || { echo "Failed to extract the archive"; exit 1; }

# Clean up the tar file
echo "Cleaning up the archive file..."
rm enron_mail_20150507.tar.gz

# Step 4: Follow step 3.2 from the SWiSSSE repo
# Assuming step 3.2 is related to processing emails or preparing the environment.
# Adding placeholder commands to demonstrate execution:
echo "Executing step 3.2 from SWiSSSE repository..."
if [ -f "scripts/process_emails.sh" ]; then
    bash scripts/process_emails.sh || { echo "Step 3.2 script execution failed"; exit 1; }
else
    echo "Expected step 3.2 script not found. Please verify the repository contents."; exit 1;
fi

# Step 5: Start the program using Docker Compose
echo "Starting the program using Docker Compose..."
docker-compose up -d --build || { echo "Docker Compose failed to start the program"; exit 1; }

# Success message
echo "Setup completed successfully! Follow step 3.3 and 3.4 from the SWiSSSE repository to run the experiments. Add the name of the docker container into experiment.py!"

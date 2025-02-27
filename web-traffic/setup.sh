#!/bin/bash
#set -e
#WARNING: THIS SCRIPT REQUIRES ROOT (SUDO)
#Do not forget to grand execution permission to this script: sudo chmod +x path/to/script/setup.sh

#----------------------------------------START OF VARIABLES----------------------------------------
#Get the name of the user running this shell script
USER_NAME=${SUDO_USER:-$(whoami)}

#Get the home directory of the user
USER_DIR=$(eval echo "~$USER_NAME")

# Path to the experiment.py file
EXPERIMENT_PATH="$USER_DIR/privacy-carbon-experiments/web-traffic/experiment.py"

#-----------------------------------------END OF VARIABLES-----------------------------------------




#----------------------------------------START OF FUNCTIONS----------------------------------------

# Function to check for root privileges (needed for package installation etc.)
check_root() {
	if [ ${EUID} -ne 0 ]; then
		echo -e "Root privileges were not granted.\nThis script needs root in order to run! Exiting."
	    	exit 1
	fi
}

check_experiment_path() {
	# Prompt the user to confirm
	echo -e "This script will execute the experiment after the setup is finished.\nThe current experiment path is set to: $EXPERIMENT_PATH"
	read -p "Is this path correct? (y/n): " response

	# Check the user's response
	case "$response" in
    	[Yy]* )
        	echo "Path confirmed. Starting execution..."
        	# Proceed with the script
        	;;
    	[Nn]* )
        	echo "Path is not correct. Please first update the path before executing. Exiting script."
        	exit 1
        	;;
    	* )
        	echo "Invalid response. Please run the script again and answer 'y' or 'n'."
        	exit 1
        	;;
	esac
}

# Function to install packages with the appropriate package manager
install_packages_with_package_manager() {
	local packages="$1"

	if command -v apt &> /dev/null; then
		echo "Detected apt (Debian/Ubuntu-based system). Installing $packages."
    	sudo apt install -y $packages
	elif command -v dnf &> /dev/null; then
    	echo "Detected dnf (Fedora-based system). Installing $packages."
    	sudo dnf install -y $packages
	elif command -v yum &> /dev/null; then
   		echo "Detected yum (CentOS/RHEL-based system). Installing $packages."
    	sudo yum install -y $packages
	elif command -v pacman &> /dev/null; then
    	echo "Detected pacman (Arch-based system). Installing $packages."
    	sudo pacman -Sy $packages
	elif command -v zypper &> /dev/null; then
    	echo "Detected zypper (openSUSE-based system). Installing $packages."
    	sudo zypper install -y $packages
	else
    	echo "No supported package manager found on this system. Exiting."
    	exit 1
	fi
}

# Function to enable Nginx and configure the firewall
enable_nginx_and_firewall() {
	echo "Enabling Nginx and configuring firewall."

	# Enable and start Nginx
	sudo systemctl enable nginx
	sudo systemctl start nginx

	# Check for active firewall
	if command -v ufw &> /dev/null; then
    	echo "Configuring firewall with UFW."
    	sudo ufw allow 'Nginx Full'
    	sudo ufw reload
	elif command -v firewall-cmd &> /dev/null; then
    	echo "Configuring firewall with Firewalld."
    	sudo firewall-cmd --permanent --add-service=http
    	sudo firewall-cmd --permanent --add-service=https
    	sudo firewall-cmd --reload
	elif command -v iptables &> /dev/null; then
    	echo "Configuring firewall with iptables."
    	sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    	sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    	sudo iptables-save > /etc/iptables/iptables.rules
	else
    	echo "No supported firewall detected. Please configure your firewall manually to support ports 80 & 443."
	fi
}

# Function to disable SELinux if enforcing
disable_SELinux() {
	# Check if the 'getenforce' command is available
	if ! command -v getenforce &> /dev/null; then
    	echo "SELinux is not installed or not available on this system."
    	# Do nothing if SELinux is not present
    	return 0
	fi

	# Get the current SELinux status
	local output
	output=$(getenforce)

	# Disable SELinux if it is enforcing
	if [ "$output" == "Enforcing" ]; then
    	sudo setenforce 0
    	echo "SELinux has been set to Permissive mode."
	fi
}

# Function to navigate to Downloads folder (and create the folder first if non-existent)
navigate_downloads() {
	mkdir -p "$USER_DIR/HTTPSCarbonExperimentDownloads"
	cd "$USER_DIR/HTTPSCarbonExperimentDownloads"
}

#-----------------------------------------END OF FUNCTIONS-----------------------------------------




#------------------------------------------START OF SCRIPT------------------------------------------

# Check for root privileges
check_root

# Check that the experiment path is correct
check_experiment_path

# Ensure we are in the home directory
cd "$USER_DIR"

# Install Nginx
install_packages_with_package_manager "nginx"

# Enable Nginx and configure firewall
enable_nginx_and_firewall

# Install pip
install_packages_with_package_manager "python3-pip"

# Install python3 virtual env

# Install curl & 7z & httrack
install_packages_with_package_manager "curl p7zip-full httrack"

# Download Wikipedia 2007 dump
navigate_downloads
echo "Downloading Wikipedia 2007 dump"
curl -s -o "wikipedia-simple-html.7z" "https://dumps.wikimedia.org/other/static_html_dumps/April_2007/simple/wikipedia-simple-html.7z"

# Extract the Wikipedia 2007 dump
mkdir -p ./wikipedia-simple-html
sudo 7za x "wikipedia-simple-html.7z" -o./wikipedia-simple-html

# Remove the .7z file
rm "wikipedia-simple-html.7z"
cd "$USER_DIR"

# Fetch the user running nginx from the nginx.conf file
NGINX_USER=$(grep -E '^user' /etc/nginx/nginx.conf | awk '{print $2}' | sed 's/;//')

# Change permissions of the Wikipedia dump so Nginx can access it
echo "Changing Wikipedia folder permissions"
sudo setfacl -R -m u:$NGINX_USER:r-x,d:u:$NGINX_USER:r-x HTTPSCarbonExperimentDownloads/wikipedia-simple-html

# Disable SELinux (if applicable)
echo "disabling SELinux temporarily if applicable"
disable_SELinux

# Download New York Times dump
echo "Downloading New York Times dump"
httrack "https://www.nytimes.com/" -O HTTPSCarbonExperimentDownloads/nytimes -r2 --robots=0 --depth=2

# Download MDN docs dump
echo "Downloading MDN docs dump"
httrack "https://developer.mozilla.org/en-US/docs/Learn" -O HTTPSCarbonExperimentDownloads/mdn_learn -r2 --robots=0 --depth=2

# Download Mastodon Blog dump
httrack "https://blog.joinmastodon.org/" -O HTTPSCarbonExperimentDownloads/mastodon_blog -r2 --robots=0 --depth=2

# Download xkcd
httrack "https://xkcd.com/" -O HTTPSCarbonExperimentDownloads/xkcd -r2 --robots=0 --depth=2

# Install OpenSSL
install_packages_with_package_manager "openssl"

# Create directory for certificate/key
cd /etc/nginx
sudo mkdir -p ssl
cd ./ssl

# Create localhost certificate config
echo "Creating localhost TLS certificate configuration"
cat > "localhost.conf" << 'EOF'
[ req ]
default_bits       = 2048
default_keyfile    = localhost.key
distinguished_name = req_distinguished_name
x509_extensions    = v3_req
prompt             = no

[ req_distinguished_name ]
CN = localhost

[ v3_req ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = localhost
EOF

# Generate private key
echo "Generating private key"
openssl genrsa -out "localhost.key" 2048

# Generate certificate
echo "Generating certificate"
openssl req -x509 -new -nodes -key "localhost.key" -sha256 -days 365 -out "localhost.crt" -config "localhost.conf"

# Change permissions of private key
echo "Changing private key permissions"
sudo chmod 600 "localhost.key"

# Create Nginx config
echo "Creating Nginx config file"
cat > /etc/nginx/nginx.conf << EOF
events {}

worker_processes auto;

http {
    server {
        listen 80;
        listen 443 ssl;
	server_name localhost;

        ssl_certificate /etc/nginx/ssl/localhost.crt;
        ssl_certificate_key /etc/nginx/ssl/localhost.key;
        ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        root $USER_DIR/HTTPSCarbonExperimentDownloads;

        location /wikipedia-simple-html/simple {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files \$uri \$uri/ =404;
        }

        location /nytimes/www.nytimes.com {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files \$uri \$uri/ =404;
        }

        location /mdn_learn/developer.mozilla.org {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files \$uri \$uri/ =404;
        }
    }
}
EOF

# Restart Nginx server
echo "Restarting Nginx server to load new changes"
sudo systemctl restart nginx

# Go back to home directory
cd "$USER_DIR"

# Load environment variables
source ~/.bashrc

echo "Creating venv using python"
python3 -m venv HTTPSCarbonExperimentDownloads/venv

# Activate the venv
echo "Activating the venv"
source $USER_DIR/HTTPSCarbonExperimentDownloads/venv/bin/activate

# Install necessary packages
pip install codecarbon matplotlib pandas requests

# Give execution permission to experiment.py
sudo chmod +x $EXPERIMENT_PATH

# Run the experiment (CHANGE PATH TO EXPERIMENT IF NECESSARY)
sudo $USER_DIR/HTTPSCarbonExperimentDownloads/venv/bin/python $EXPERIMENT_PATH

#-------------------------------------------END OF SCRIPT-------------------------------------------
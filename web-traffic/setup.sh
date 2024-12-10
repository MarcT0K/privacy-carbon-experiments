#!/bin/bash
set -e

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
    local output
    output=$(getenforce)

    if [ "$output" == "Enforcing" ]; then
        sudo setenforce 0
    fi
}

# Function to navigate to Downloads folder (and create the folder first if non-existent)
navigate_downloads() {
	mkdir -p "$HOME/Downloads"	
	cd "$HOME/Downloads"
	fi
}

# Ensure we are in the home directory
cd "$HOME"

# Install Nginx
install_packages_with_package_manager "nginx"

# Enable Nginx and configure firewall
enable_nginx_and_firewall

# Install pip
install_packages_with_package_manager "python3-pip"

# Install Requests
echo "Installing Requests Python library"
pip3 install requests

# Install curl & 7z & httrack
install_packages_with_package_manager "curl p7zip httrack"

# Download Wikipedia 2007 dump
navigate_downloads
echo "Downloading Wikipedia 2007 dump"
curl -s -o "wikipedia-simple-html.7z" "https://dumps.wikimedia.org/other/static_html_dumps/April_2007/simple/wikipedia-simple-html.7z"

# Extract the Wikipedia 2007 dump
7z x "wikipedia-simple-html.7z"

# Remove the .7z file
rm "wikipedia-simple-html.7z"
cd "$HOME"

# Change permissions of the Wikipedia dump so Nginx can access it
sudo setfacl -R -m u:nginx:rx -m d:u:nginx:rx Downloads/wikipedia-simple-html

# Disable SELinux (if applicable)
disable_SELinux

# Download New York Times dump
httrack "https://www.nytimes.com/" -O Downloads/nytimes -r2 --robots=0

# Download GitHub Chromium-Project dump
httrack "https://github.com/chromium/chromium" -O Downloads/github -r2 --robots=0

# Download MDN docs dump
httrack "https://developer.mozilla.org/en-US/docs/Learn" -O Downloads/mdn_learn -r2 --robots=0

# Download amazon dump
httrack "https://www.amazon.nl/" -O Downloads/amazon -r2 --robots=0

# Install OpenSSL
install_packages_with_package_manager "openssl"

# Create directory for certificate/key
cd /etc/nginx
sudo mkdir -p ssl
cd ./ssl

# Create localhost certificate config
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
openssl genrsa -out "localhost.key" 2048

# Generate certificate
openssl req -x509 -new -nodes -key "localhost.key" -sha256 -days 365 -out "localhost.crt" -config "localhost.conf"

# Change permissions of private key
sudo chmod 600 "localhost.key"

# Create Nginx config
cat > /etc/nginx/nginx.conf << "EOF"
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

        root $HOME/Downloads;

        location /wikipedia-simple-html/simple {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files $uri $uri/ =404;
        }

        location /nytimes/www.nytimes.com {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files $uri $uri/ =404;
        }

        location /github/github.com {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files $uri $uri/ =404;
        }

        location /mdn_learn/developer.mozilla.org {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files $uri $uri/ =404;
        }

        location /amazon/www.amazon.nl {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files $uri $uri/ =404;
        }
    }
}
EOF

# Restart Nginx server
sudo systemctl restart nginx


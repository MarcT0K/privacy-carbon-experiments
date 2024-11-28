#IMPORTANT: THIS SCRIPT ONLY RUNS ON LINUX SYSTEMS WITH APT/DNF/YUM/PACMAN/ZYPPER AS PACKAGE MANAGER

#!/bin/bash
set -e

install_packages_with_package_manager() {
	local packages = "$1"

	if command -v apt &> /dev/null; then
		echo "Detected apt (Debian/Ubuntu-based system)."
		echo "Installing $packages."
		sudo apt install -y $packages
	elif command -v dnf &> /dev/null; then
		echo "Detected dnf (Fedora-based system)."
		echo "Installing $packages."
		sudo dnf install -y $packages
	elif command -v yum &> /dev/null; then
		echo "Detected yum (CentOS/RHEL-based system)."
		echo "Installing $packages."
		sudo yum install -y $packages
	elif command -v pacman &> /dev/null; then
		echo "Detected pacman (Arch-based system)."
		echo "Installing $packages."
		sudo pacman -Sy $packages
	elif command -v zypper &> /dev/null; then
		echo "Detected zypper (openSUSE-based system)."
		echo "Installing $packages."
		sudo zypper install -y $packages
	else
		echo "No supported package manager found on this system. (system unsupported)"
		exit 1
	fi
}

enable_nginx_and_firewall() { #TODO: make ufw and firewalld check such that we do not hardcode by packagemanager
    	if command -v apt &> /dev/null; then
		echo "Enabling Nginx and configuring firewall."
		sudo systemctl enable nginx
		sudo systemctl start nginx
		sudo ufw allow 'Nginx Full'
		sudo ufw reload
    	elif command -v dnf &> /dev/null; then
		echo "Enabling Nginx and configuring firewall."
		sudo systemctl enable nginx
		sudo systemctl start nginx
		sudo firewall-cmd --permanent --add-service=http
		sudo firewall-cmd --permanent --add-service=https
		sudo firewall-cmd --reload
    	elif command -v yum &> /dev/null; then
		echo "Enabling Nginx and configuring firewall."
		sudo systemctl enable nginx
		sudo systemctl start nginx
		sudo firewall-cmd --permanent --add-service=http
		sudo firewall-cmd --permanent --add-service=https
		sudo firewall-cmd --reload
    	elif command -v pacman &> /dev/null; then
		echo "Enabling Nginx and configuring firewall."
		sudo systemctl enable nginx
		sudo systemctl start nginx
		sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
		sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
		sudo iptables-save > /etc/iptables/iptables.rules
    	elif command -v zypper &> /dev/null; then
		echo "Enabling Nginx and configuring firewall."
		sudo systemctl enable nginx
		sudo systemctl start nginx
		sudo firewall-cmd --permanent --add-service=http
		sudo firewall-cmd --permanent --add-service=https
		sudo firewall-cmd --reload
    	else
    		echo "System unsupported."
    	fi
}

disable_SELinux() {
	local output=$(getenforce)
	
	if [output == "Enforcing"]; then
		setenforce 0
}

#ensure we are in root directory
cd

#install Nginx
install_packages_with_package_manager "nginx"

#enable Nginx & configure firewall (& reload firewall if necessary)
enable_nginx_and_firewall

#install pip
install_packages_with_package_manager "python3-pip"

#install Requests
echo "Installing Requests library"
pip3 install requests

#install curl & 7z
install_packages_with_package_manager "curl p7zip"

#download WikiMedia 2007 dump
cd ~/Downloads
echo "downloading WikiMedia 2007 dump"
curl -s -o "wikipedia-simple-html.7z" "https://dumps.wikimedia.org/other/static_html_dumps/April_2007/simple/wikipedia-simple-html.7z"

#extract the WikiMedia 2007 dump
7z x "wikipedia-simple-html.7z"

#remove the .7z file
rm "wikipedia-simple-html.7z"
cd

#change permissions of dump such that nginx can serve the dump
sudo setfacl -R -m u:nginx:rx -m d:u:nginx:rx Downloads/wikipedia-simple-html
disable_SELinux

#install openssl
install_packages_with_package_manager "openssl"

#create directory for certificate/key
cd "/etc/nginx"
mkdir "ssl"

#create localhost certificate config
cd "./ssl"

cat > "localhost.conf" << EOF
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

#generate private key
openssl genrsa -out "localhost.key" 2048

#generate certificate
openssl req -x509 -new -nodes -key "localhost.key" -sha256 -days 365 -out "localhost.crt" -config "localhost.conf"

#change permissions of private key
sudo chmod 600 "localhost.key"

#create nginx config
cd ".."

local ninx_config = 
"events {}

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

        root $HOME/Downloads/wikipedia-simple-html/simple;
        index index.html;

        location / {
            autoindex on;
            autoindex_exact_size off;
            autoindex_localtime on;
            try_files $uri $uri/ =404;
        }
    }
}"

echo nginx_config > "nginx.conf"

#restart nginx server
sudo systemctl restart nginx







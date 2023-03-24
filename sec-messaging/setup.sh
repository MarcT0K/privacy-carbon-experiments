#!/bin/bash
set -e

sudo apt install python3 python3-pip wget

pip3 install -r requirements.txt

# Enron Dataset
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -O enron.tar.gz
tar xzvf enron.tar.gz
rm enron.tar.gz
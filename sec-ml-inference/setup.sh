#!/bin/bash
set -e

sudo apt install python3 python3-pip

pip install -U pip wheel setuptools
pip3 install -r requirements.txt

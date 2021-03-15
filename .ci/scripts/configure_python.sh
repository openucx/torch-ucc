#!/bin/bash -eEx
set -o pipefail

alternatives --set python /usr/bin/python3
pip3 install --user --upgrade setuptools wheel

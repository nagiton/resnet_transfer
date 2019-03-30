#!/bin/bash

set -euxo pipefail

sudo apt-get install software-properties-common
sudo add-apt-repository ppa:webupd8team/atom
sudo apt-get update
sudo apt-get install atom

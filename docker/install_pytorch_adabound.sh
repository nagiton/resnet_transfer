#!/bin/bash

set -euxo pipefail

sudo apt-get install python3-pip
sudo pip install https://download.pytorch.org/whl/cu80/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
sudo pip install torchvision

sudo pip install visdom
pip install git+https://github.com/pytorch/tnt.git@464aa492716851a6703b90c0c8bb0ae11f8272da

"""
Shell script to set up GPU when booting up on Google Cloud Platform
Installs:
Anaconda3, Jupyter Notebook, cuda 9.1, Tensorflow, Keras, and AWS CLI
Base code credits to https://github.com/joannasys

To run:
Setup following instructions here: https://medium.com/@howkhang/ultimate-guide-to-setting-up-a-google-cloud-machine-for-fast-ai-version-2-f374208be43
Copy and paste the entire chunk below into your remote terminal on startup.
After everything runs (~5minutes) there is a reboot at the end of the script (necessary)

SSH in again with:
gcloud compute ssh yourinstancename

and we're in!
"""

#!/bin/bash

DEBIAN_FRONTEND=noninteractive

sudo rm /etc/apt/apt.conf.d/*.*
sudo apt update
mkdir downloads
cd downloads
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh -b
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc
conda env update
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
sudo ufw allow 8888/tcp
sudo apt -y install qtdeclarative5-dev qml-module-qtquick-controls
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
cd ~/downloads/
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt install cuda -y
wget http://files.fast.ai/files/cudnn-9.1-linux-x64-v7.tgz
tar xf cudnn-9.1-linux-x64-v7.tgz
sudo cp cuda/include/*.* /usr/local/cuda/include/
sudo cp cuda/lib64/*.* /usr/local/cuda/lib64/
#pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
sudo apt-get install python-dev python-pip libcupti-dev -y
conda install -c anaconda tensorflow-gpu -y
conda install -c anaconda keras-gpu
conda install -c conda-forge awscli
sudo reboot

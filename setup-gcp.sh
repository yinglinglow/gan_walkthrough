"""
Shell script to set up GPU when booting up on Google Cloud Platform
Installs:
Anaconda3, Jupyter Notebook, cuda 9.1, Tensorflow, Keras, and AWS CLI
Base code credits to https://github.com/joannasys

To run:
Setup following instructions here: https://medium.com/@howkhang/ultimate-guide-to-setting-up-a-google-cloud-machine-for-fast-ai-version-2-f374208be43

SSH in with:
gcloud compute ssh yourinstancename

Copy and paste the entire chunk below into your remote terminal on startup.

and we're ready!
"""

#!/bin/bash
conda install -c anaconda keras-gpu
conda install -c conda-forge awscli

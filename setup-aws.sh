"""
Shell script to set up GPU when booting up on AWS EC2

Set up your EC2 using the ami 'ami-ccba4ab4' by Adrian Rosebrock on:
https://www.pyimagesearch.com/2017/09/20/pre-configured-amazon-aws-deep-learning-ami-with-python/

To run:
Copy and paste each of these parts into the terminal

I tried to automate this... but it didn't work :( 
If there are any suggestions at all please let me know!!

"""

#!/bin/bash

# reboot now, and SSH in again with 'ssh -i yourpemfile.pem ubuntu@ec2-xx-xxx-xxx-xxx'
# OR reinstall NVIDIA driver everytime you launch/reboot your instance... not fun
sudo reboot

# part 1
cd installers
sudo ./NVIDIA-Linux-x86_64-375.26.run --silent
cd ..

# part 2
workon dl4cv

# part 3
pip3 install awscli
pip3 install pandas
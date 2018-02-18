"""
Shell script to set up and run the model

To run:
Copy and paste each of these parts into the terminal

"""

git clone https://github.com/yinglinglow/gan_walkthrough.git
cd gan_walkthrough
mkdir gan
mkdir gan_models

# change your variables accordingly if necessary
export XTRAIN=X_train_56_1700.pkl
export CODE=DCGAN_160218_final
export DATE=180218

# if you uploaded your training file onto AWS S3, import it now, else skip this chunk
# paste your AWS keys in (DO NOT SAVE YOUR KEYS HERE/ PUSH IT TO GITHUB)
# export AWS_ACCESS_KEY_ID=xxxxxxxxxxxxxxx
# export AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxx
# aws s3 cp s3://gan-project/$XTRAIN .

# open tmux, and run the model
tmux
python3 $CODE.py --output_dir=wgan


"""
Results are saved in the directories: gan
Models are saved in the directories: gan_model
    
To save images and models to AWS directly:
aws s3 cp -r gan/ s3://yourbucketname/
aws s3 cp -r gan_models/ s3://yourbucketname/

To save files from instance to local:

AWS: scp -i yourpemfile.pem -r ubuntu@ec2-xx-xxx-xxx-xxx.us-west-2.compute.amazonaws.com:~/gan_walkthrough/gan .
GCP: gcloud compute scp instance-1:gan_walkthrough/gan/ .

"""
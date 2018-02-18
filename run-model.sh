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
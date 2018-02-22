# GAN Walkthrough (DCGAN & WGAN-GP)

This is a walkthrough for people new to deep learning and GAN, to learn about and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

For the full blogpost, refer to: [https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough](https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough)

Full credits go to [Rowel Atienza](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py) for DCGAN code and [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) for WGAN-GP code.

1. [Obtaining Dataset](#obtaining-dataset)
2. [Cleaning Dataset](#cleaning-dataset)
3. [Set up Cloud Platform](#set-up-cloud-platform)
4. [Running the Model](#running-the-model)
5. [Results](#results)

## Goal

Generate new brand logos from logos designed by humans

## Obtaining Dataset

Download the folder of pictures I used, from `logos_originals_1367.zip`.

<img width="400" alt="logo_originals" src="https://user-images.githubusercontent.com/21985915/36466353-aefc2484-1714-11e8-94b2-b35364527a01.png">


## Cleaning Dataset

```bash
# to center crop and resize the images
python3 1_5_resize_centre.py --path=/Users/xxx/to_resize/ --size=56

# to convert all pictures to one big array and pickle it
python3 1_6_resize_to_array.py --path=/Users/xxx/resized/ --height=56 --target_path=/Users/xxx/ --augment=True

# optional: to upload to AWS S3 using the AWS CLI
aws s3 cp /Users/xxx/X_train_56_1700.pkl s3://yourbucketname/
```

## Set up cloud platform (if you do not have a GPU)
_Work in progress - DCGAN works fine on both AWS and GCP but WGAN can only run on AWS :(_


__For AWS__<br>
Set up your EC2 (p2x.large) instance using the ami 'ami-ccba4ab4' by Adrian Rosebrock on: 
[https://www.pyimagesearch.com/2017/09/20/pre-configured-amazon-aws-deep-learning-ami-with-python/](https://www.pyimagesearch.com/2017/09/20/pre-configured-amazon-aws-deep-learning-ami-with-python/)

Then, install AWSCLI and pandas.

```bash
pip3 install awscli
pip3 install pandas
```

__For GCP__<br>
Set up your gcloud compute instance using this: [https://medium.com/@howkhang/ultimate-guide-to-setting-up-a-google-cloud-machine-for-fast-ai-version-2-f374208be43](https://medium.com/@howkhang/ultimate-guide-to-setting-up-a-google-cloud-machine-for-fast-ai-version-2-f374208be43)

Then, install AWSCLI and Keras.

```bash
conda install -c anaconda keras-gpu
conda install -c conda-forge awscli
```

## Running the Model

```bash
# git clone everything in
git clone https://github.com/yinglinglow/gan_walkthrough.git
cd gan_walkthrough
mkdir gan
mkdir gan_models

# open tmux
tmux

# change your variables accordingly if necessary
export XTRAIN=X_train_56_1366.pkl
export CODE=WGAN_180218_final
export DATE=210218

# run the model
python3 $CODE.py
```

__To save result files to AWS S3 directly__<br>
aws s3 cp gan/* s3://yourbucketname/<br>
aws s3 cp gan_models/* s3://yourbucketname/

__To save result files to your local computer__<br>
Run the below commands from your LOCAL terminal!!

```bash
# for AWS
scp -i yourpemfile.pem -r ubuntu@ec2-xx-xxx-xxx-xxx.us-west-2.compute.amazonaws.com:~/gan_walkthrough/gan/* .<br>
scp -i yourpemfile.pem -r ubuntu@ec2-xx-xxx-xxx-xxx.us-west-2.compute.amazonaws.com:~/gan_walkthrough/gan_models/* .

# for GCP
gcloud compute scp yourinstancename:gan_walkthrough/gan/* .
```

## Results

__1) DCGAN (56x56)__<br>
Epoch: 3000 <br>
<img src='https://user-images.githubusercontent.com/21985915/36361986-a2bd0bac-156b-11e8-9d07-fb39dc348440.png' width="200"><br><br>

__2) WGAN-GP (56x56)__<br>
Epoch: 2000<br>
<img src='https://user-images.githubusercontent.com/21985915/36361988-a320d7e0-156b-11e8-961f-13719a3c1088.png' width="200"><br>
Epoch: 2500<br>
<img src='https://user-images.githubusercontent.com/21985915/36361989-a351681a-156b-11e8-9220-c514a66e1b1d.png' width="200"><br>
__3) WGAN-GP (112x112)__<br>
Epoch: 2500<br>
<img src='https://user-images.githubusercontent.com/21985915/36516668-1fc93008-17ba-11e8-8c44-7a66ebf8cbd5.png' width="200">

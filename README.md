# GAN Walkthrough (DCGAN & WGAN-GP)

This is a walkthrough for people new to deep learning and GAN, to learn about and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

For the full blogpost, refer to: [https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough](https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough)

Full credits go to [Rowel Atienza](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py) for DCGAN code and [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) for WGAN-GP code.

1. [Obtaining Dataset](#obtaining-dataset)
2. [Cleaning Dataset](#cleaning-dataset)
3. [Running the Model](#running-the-model)
4. [Results](#results)

## Goal

Generate new brand logos from logos designed by humans

## Obtaining Dataset
There are 3 different ways to obtain your starting images - I recommend method 2 (scraping from Google Images).

__1) Scrape 80,000 logos from Wikipedia__

Scrape all the links for the images from [Wikipedia page](https://commons.wikimedia.org/wiki/Category:Unidentified_logos) and save it to and items.csv file, using:
```bash
scrapy runspider 1_1_wikispider.py -o items.csv -t csv
```

To download all the images, use:
```bash
python3 1_2_downloading_wiki_pics.py --filename=items.csv --local=True
```

<img src='https://user-images.githubusercontent.com/21985915/36363186-dd9d4e80-1575-11e8-98d5-aa797107ee4c.png' width=400>

__2) Scrape 2,000 logos scraped from Google Images__

Use this: [https://github.com/hardikvasa/google-images-download](https://github.com/hardikvasa/google-images-download) from Hardik Vasa

Use various keywords such as _'logo'_, _'logo circle'_, _'logo simple'_, _'logo vector'_, etc. Be sure to look through your logos manually and ensure that they are of good quality. If you need to split logos arranged in a grid into individual photos, use:
```bash
python3 1_3_split_pic.py --filename=abc.jpeg --col=3 --row=2
```

Alternatively, you can simply download the folder of pictures I used, from `logos_originals_1700.zip`.

<img src='https://user-images.githubusercontent.com/21985915/36361926-0df0aa24-156b-11e8-964e-42cb13c0de9c.png' width=400>


__3) Download 800 logos from Font Awesome (black and white)__

Download from here: https://fontawesome.com/

Unzip and navigate into advanced-options, and raw-svg.

This contains all the svg files (meaning they are stored as vectors instead of pixels). To convert them into png files, use:
```bash
python3 1_4_convert_svg_png.py --path=/Users/xxx/svgtopng/
```

<img src='https://user-images.githubusercontent.com/21985915/36363188-e31f908e-1575-11e8-9612-1b87209f1a81.png' width=400>

## Cleaning Dataset

__1) Center crop and resize to 56 x 56__

To center crop and resize them, use:
```bash
python3 1_5_resize_centre.py --path=/Users/xxx/to_resize/
```

__2) Append to array__

To convert all pictures to one big array and pickle it, use:
```bash
python3 1_6_resize_to_array.py --path=/Users/xxx/resized/ --height=56 --target_path=/Users/xxx/ to_augment=True
```

__3) Upload array to S3__

To upload to AWS S3 using AWS CLI, use:
```bash
aws s3 cp /Users/xxx/X_train_56_1700.pkl s3://gan-project/
```

# Set up cloud platform (if you do not have a GPU)

Use `setup-aws.sh` for AWS EC2, or `setup-gcp.sh` for Google Cloud Platform.
*In progress - DCGAN works fine on GCP but WGAN has some issues, potentially due to installation problems :(

## Running the Model

__1) DCGAN__

Use `run-model.sh` to run the model. 
Change the variables accordingly to whichever model or XTRAIN set you are using.

```bash
git clone https://github.com/yinglinglow/gan_walkthrough.git
cd gan_walkthrough
mkdir gan
mkdir gan_models

# open tmux
tmux

# change your variables accordingly if necessary
export XTRAIN=X_train_56_1700.pkl
export CODE=WGAN_180218_11am
export DATE=180218

# run the model
python3 $CODE.py
```

## Results


__1) DCGAN__

<img src='https://user-images.githubusercontent.com/21985915/36361986-a2bd0bac-156b-11e8-9d07-fb39dc348440.png' width=200>

Epoch: 3000

__2) WGAN-GP__

<img src='https://user-images.githubusercontent.com/21985915/36361988-a320d7e0-156b-11e8-961f-13719a3c1088.png' width=200>

Epoch: 2000

<img src='https://user-images.githubusercontent.com/21985915/36361989-a351681a-156b-11e8-9220-c514a66e1b1d.png' width=200>

Epoch: 2500

<img src='https://user-images.githubusercontent.com/21985915/36361990-a3885618-156b-11e8-9975-dc16a7ca323a.png' width=200>

Epoch: 3000
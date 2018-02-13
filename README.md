# GAN Walkthrough

This is a walkthrough for people new to deep learning and GAN, to learn about and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

For the full blogpost, refer to: [https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough](https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough)

Full credits go to [Rowel Atienza](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py) for DCGAN and [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) for WGAN-GP code.

1. [Obtaining Dataset](#obtaining-dataset)
2. [Cleaning Dataset](#cleaning-dataset)
3. [Developing Model](#developing-model)
4. [Results](#results)

## Goal

Generate new brand logos from logos designed by humans




## Obtaining Dataset
There are 3 different ways to obtain your starting images - I recommend method 2 (scraping from Google Images).

__1) Scrape 80,000 logos from Wikipedia__

Scrape all the links for the images from [Wikipedia page](https://commons.wikimedia.org/wiki/Category:Unidentified_logos), using:
```bash
1_1_1_wikispider.py
```

Edit from line 54/55 for csv_filename and bucketname if necessary.
To download, use:
```bash
1_1_2_downloading_wiki_pics.py
```

__2) Scrape 2,000 logos scraped from Google Images__

Use this: [https://github.com/hardikvasa/google-images-download](https://github.com/hardikvasa/google-images-download) from Hardik Vasa

Use various keywords such as _'logo'_, _'logo circle'_, _'logo simple'_, _'logo vector'_, etc. Be sure to look through your logos manually and ensure that they are of good quality. If you need to split logos arranged in a grid into individual photos, use the _split_pic_ function from `1_1_3_split_pics_svg.py`.

Alternatively, you can simply download the folder of pictures I used, from `logos_originals_1775.zip`.


__3) Download 800 logos from Font Awesome (black and white)__

Download from here: https://fontawesome.com/

Unzip and navigate into advanced-options, and raw-svg.

This contains all the svg files (meaning they are stored as vectors instead of pixels). To convert them into png files, use the _convert_svg_png_ function from `1_1_3_split_pics_svg.py`.

## Cleaning Dataset

__1) Center crop and resize to 56 x 56__

To center crop and resize them, use the _resize_centre_ function from `1_1_3_split_pics_svg.py`.

## Developing Model

__1) DCGAN__
Use:

```bash
python3 
```

__2) WGAN-GP__
Use:

```bash
python3 
```

## Results

__1) DCGAN__


__2) WGAN-GP__
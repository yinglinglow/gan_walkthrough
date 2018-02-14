# GAN Walkthrough (DCGAN & WGAN-GP)

This is a walkthrough for people new to deep learning and GAN, to learn about and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

For the full blogpost, refer to: [https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough](https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough)

Full credits go to [Rowel Atienza](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py) for DCGAN code and [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) for WGAN-GP code.

1. [Obtaining Dataset](#obtaining-dataset)
2. [Cleaning Dataset](#cleaning-dataset)
3. [Developing Model](#developing-model)
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

__2) Scrape 2,000 logos scraped from Google Images__

Use this: [https://github.com/hardikvasa/google-images-download](https://github.com/hardikvasa/google-images-download) from Hardik Vasa

Use various keywords such as _'logo'_, _'logo circle'_, _'logo simple'_, _'logo vector'_, etc. Be sure to look through your logos manually and ensure that they are of good quality. If you need to split logos arranged in a grid into individual photos, use:
```bash
python3 1_3_split_pic.py --filename=abc.jpeg --col=3 --row=2
```

Alternatively, you can simply download the folder of pictures I used, from `logos_originals_1700.zip`.


__3) Download 800 logos from Font Awesome (black and white)__

Download from here: https://fontawesome.com/

Unzip and navigate into advanced-options, and raw-svg.

This contains all the svg files (meaning they are stored as vectors instead of pixels). To convert them into png files, use:
```bash
python3 1_4_convert_svg_png.py --path=/Users/xxx/svgtopng/
```

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
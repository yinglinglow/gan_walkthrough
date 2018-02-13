# GAN Walkthrough

This is a walkthrough for people new to deep learning and GAN to learn about GAN, and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

For the full blogpost, refer to: [https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough](https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough)

## Goals

Generate new brand logos from logos designed by humans

## Obtaining Dataset

__1) Scrap 80,000 logos from Wikipedia__

Scrap all the links for the images from wikipedia page, using:
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

Use various keywords such as 'logo', 'logo circle', 'logo simple', 'logo vector', etc
Be sure to look through your logos manually and ensure that they are of good quality.

Alternatively, you can download the folder of pictures I used, here: __logos_originals_1775.zip__

Use the _split_pic_ function from `1_1_3_split_pics_svg.py`.

__3) 800 logos downloaded from Font Awesome (for black and white logos)__

Download from here: https://fontawesome.com/

Unzip and navigate into advanced-options, and raw-svg.

This contains all the svg files (meaning they are stored as vectors instead of pixels). 

To convert them into png files, use the _convert_svg_png_ function from `1_1_3_split_pics_svg.py`.

## Cleaning Dataset

__Center crop and Resize to 56 x 56__

To center crop and resize them, use the _resize_centre_ function from `1_1_3_split_pics_svg.py`.

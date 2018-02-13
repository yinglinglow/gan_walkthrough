# GAN Walkthrough

__This is a walkthrough for:__ people new to deep learning and GAN to learn about GAN, and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

__Goals__

Generate new brand logos from logos designed by humans

__Dataset__

1) 80,000 logos scraped from Wikipedia (for volume)
_To be frank the results using 80k images were not fantastic - so if it is too much trouble, skip downloading this and just scrape from Google Images._

To scrap, use _1_1_1_wikispider.py_, which saves all the links into 'items.csv' in a tab delimited csv.

```bash
scrapy runspider wikispider.py -o items.csv -t csv
```
Subsequently, there are a few options (I recommend the third!!):
1) download the files to your local drive and store them there
2) download the files directly into AWS (saves space on your local drive, but slow)
3) download the files to your local drive and upload to AWS via awscli (much faster)

Edit from line 54/55 for csv_filename and bucketname if necessary.
To download, use:
```bash
1_1_2_downloading_wiki_pics.py
```


2) 2,000 logos scraped from Google Images (for quality)
Use this: [https://github.com/hardikvasa/google-images-download](https://github.com/hardikvasa/google-images-download) from Hardik Vasa
Use various keywords such as 'logo', 'logo circle', 'logo simple', 'logo vector', etc
Be sure to look through your logos manually and ensure that they are of good quality.

Alternatively, you can download the folder of pictures I used, here: 


3) 800 logos downloaded from Font Awesome (for black and white logos)
Download from here: https://fontawesome.com/, unzip and navigate into advanced-options, and raw-svg.
This contains all the svg files (meaning they are stored as vectors instead of pixels). 
To convert them into png files, 

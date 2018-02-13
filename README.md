# GAN Walkthrough

__This is a walkthrough for:__ people new to deep learning and GAN to learn about GAN, and be able to run their own GAN. Disclaimer: All of the below is purely for educational purposes!

For the full blogpost, refer to: [https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough](https://www.yinglinglow.com/blog/2018/02/13/GAN-walkthrough)


__Goals__

Generate new brand logos from logos designed by humans

__Dataset__

1) Scrap 80,000 logos from Wikipedia

Edit from line 54/55 for csv_filename and bucketname if necessary.
To download, use:
```bash
1_1_2_downloading_wiki_pics.py
```

2) Scrap 2,000 logos scraped from Google Images
Use this: [https://github.com/hardikvasa/google-images-download](https://github.com/hardikvasa/google-images-download) from Hardik Vasa
Use various keywords such as 'logo', 'logo circle', 'logo simple', 'logo vector', etc
Be sure to look through your logos manually and ensure that they are of good quality.


Alternatively, you can download the folder of pictures I used, here: 



3) 800 logos downloaded from Font Awesome (for black and white logos)
Download from here: https://fontawesome.com/, unzip and navigate into advanced-options, and raw-svg.
This contains all the svg files (meaning they are stored as vectors instead of pixels). 
To convert them into png files, 

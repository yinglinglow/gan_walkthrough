"""
This function takes an image and splits it into user-defined number of rows and columns
To run: e.g. split_pic('abc.jpg', 3, 2)
"""

def split_pic(filename, col, row):
    from PIL import Image
    filename = filename
    col = col
    row = row

    im = Image.open(filename)
    pic_width = im.width
    pic_height = im.height

    col_width = (pic_width/col)
    col_height = (pic_height/row)

    for width in range(col):
        for height in range(row):
            current_w = col_width*width
            current_h = col_height*height
            im1 = im.crop(box=(current_w, current_h, current_w+col_width, current_h+col_height))
            name = f"{filename[:-4]}_{width}_{height}{filename[-4:]}"
            
            im1.save(name)

# split_pic('abc.png', 5, 5)

"""
This function takes all svg images in the path, converts them to png, and saves them in a subfolder 'png'
To run: e.g. convert_svg_png('/Users/xxx/svgtopng')
"""

def convert_svg_png(path):
    from cairosvg import svg2png
    import os

    # check if directory to save png images into exists, if not create it
    directory = 'png/'
    if not os.path.exists(directory): 
        os.makedirs(directory)

    listing = [f for f in os.listdir(path) if f.endswith('.svg')]

    for file_ in listing:
        target = directory + file_[:-4] + '.png'
        svg2png(url=file_, write_to=target)

# convert_svg_png('abc.svg')

"""
This function resizes and centre crops all images in the path, and saves them in the subfolder 'resized'
To run: e.g. resize_centre('/Users/xxx/to_resize')
"""


def resize_centre(path):
    import os
    from PIL import Image, ImageOps

    # check if directory to save resized images into exists, if not create it
    directory = 'resized/'
    if not os.path.exists(directory): 
        os.makedirs(directory)

    listing = [f for f in os.listdir(path) if not f.startswith('.')]
    i = 0
    height = 56

    for file_ in listing:
        i += 1  
        im = Image.open(path + file_)
        im_resized = ImageOps.fit(im, (height, height))
        filename = directory + file_
        im_resized.save(filename)

# resize_centre('to_resize')
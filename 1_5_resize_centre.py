
"""
This function resizes and centre crops all images in the path, and 
saves them into another folder with named: path_height

To run: 
python3 1_5_resize_centre.py --path=/Users/xxx/to_resize/
"""

def remove_transparency(im, bg_colour=(255, 255, 255)):
    """ Support function to convert transparency layer to white colour """
    from PIL import Image
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def resize_centre(path):
    import os
    from PIL import Image, ImageOps

    i = 0
    height = 56

    # check if directory to save resized images into exists, if not create it
    directory = path + '_' + str(height) + '/'
    if not os.path.exists(directory): 
        os.makedirs(directory)

    listing = [f for f in os.listdir(path) if f.split('.')[-1] in ['jpg', 'png', 'jpeg']]

    # check that path ends with '/'
    if path.endswith('/'):
        pass
    else:
        path = path + '/'

    for file_ in listing:
        i += 1  
        im = Image.open(path + file_)
        im = remove_transparency(im).convert('RGB')
        im = ImageOps.fit(im, (height, height))
        filename = directory + file_
        im.save(filename)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to retrieve images from")
    args = parser.parse_args()

    path = args.path

    resize_centre(path)
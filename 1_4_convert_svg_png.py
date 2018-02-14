"""
This function takes all svg images in the path, converts them to png, 
and saves them in a subfolder 'png'.
To run: 
python3 1_4_convert_svg_png.py --path=/Users/xxx/svgtopng/
"""

def convert_svg_png(path):
    from cairosvg import svg2png
    import os

    # check if directory to save png images into exists, if not create it
    directory = 'png/'
    if not os.path.exists(directory): 
        os.makedirs(directory)

    # check that path ends with '/'
    if path.endswith('/'):
        pass
    else:
        path = path + '/'

    listing = [f for f in os.listdir(path) if f.endswith('.svg')]

    for file_ in listing:
        target = directory + file_[:-4] + '.png'
        svg2png(url=file_, write_to=target)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to retrieve svg images from")
    args = parser.parse_args()

    path = args.path

    convert_svg_png(path)
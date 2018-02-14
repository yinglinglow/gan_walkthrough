"""
This function takes an image, splits it into user-defined number of rows and columns,
and saves it in the current folder.
To run:
python3 1_3_split_pic.py --filename=abc.jpeg --col=3 --row=2
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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", required=True, help="Name of image to split")
    parser.add_argument("--col", required=True, help="Number of columns to split image into", type=int)
    parser.add_argument("--row", required=True, help="Number of rows to split image into", type=int)
    args = parser.parse_args()

    filename = args.filename
    col = args.col
    row = args.row

    split_pic(filename, col, row)
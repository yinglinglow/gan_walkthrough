"""
This function takes all images in the path, appends it to an array and 
saves it as a pickle file to the target path.

Optional argument: image augmentation to increase number of images to ~2000 if 
original number of images is at least 200.

Returns the X_train array.

To run:
python3 1_6_resize_to_array.py --path=/Users/xxx/resized/ --height=56 --target_path=/Users/xxx/ --augment=True
"""

import os
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# fix random seed for reproducibility
np.random.seed(0) 

# helper function to perform image augmentation to increase number of images to ~2000.
def augment(X_train):
    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    datagen.fit(X_train)

    X_train_partial_list = []
    for img_batch in datagen.flow(X_train, batch_size=64):
        for i in range( int(2000/len(X_train)) ):
            X_train_partial_list.append(img_batch)
        break
    array_ = np.concatenate(X_train_partial_list)
    print(array_.shape[0])
    return array_


def resize_crop_array(path, height, target_path, to_augment=False):
    # check that paths end with '/'
    if path.endswith('/'):
        pass
    else:
        path = path + '/'
        
    if target_path.endswith('/'):
        pass
    else:
        target_path = target_path + '/'

    # initialise variables
    i = 0
    array_list = []

    # load names of images
    listing = [f for f in os.listdir(path) if not f.startswith('.')]

    # loop through each image
    for file_ in listing:
        i += 1  
        im = Image.open(path + file_) # open file
        array_list.append(np.array(im).flatten()) # flatten array
    
    # shuffle full array, reshape
    immatrix = np.array(array_list)
    np.random.shuffle(immatrix)
    X_train = immatrix.reshape(immatrix.shape[0], height, height, 3)
    X_train = X_train.astype('float32')
    print(X_train.shape)
    
    # augment images if required
    if to_augment == True:
        assert len(X_train) > 200, "Number of base images should be at least 200"
        X_train = augment(X_train)
        print(X_train.shape)
    
    # pickle
    filename = 'X_train_' + str(height) + '_'
    X_train.dump(target_path + filename + str(i) + '.pkl')
    print('array saved: ' + target_path + filename + str(i) + '.pkl' + '_' + str(X_train.shape[0]))

    return X_train



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to retrieve images from")
    parser.add_argument("--height", required=True, help="Path to retrieve images from", type=int)
    parser.add_argument("--target_path", "-tp", required=True, help="Path to save array to")
    parser.add_argument("--augment", "-aug", required=False, help="True to augment to around 2000 images")
    args = parser.parse_args()

    path = args.path
    height = args.height
    target_path = args.target_path
    to_augment = args.augment

    resize_crop_array(path, height, target_path, to_augment=False)
"""
changed losses, up to 112
"""
"""
Base code from https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

Optimisation done for:
56 x 56 RGB images

Import pre-processed image array saved locally

To run:
mkdir gan
mkdir gan_models
tmux
export XTRAIN=X_train_56_1700.pkl
export CODE=WGAN_190218_final
export DATE=190218
python3 $CODE.py
"""

# necessary when running on AWS EC2
import matplotlib
matplotlib.use('Agg')

# import requirements
import os
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import pandas as pd

# fix random seed for reproducibility
np.random.seed(0) 

# fix variables
BATCH_SIZE = 64 # the number of images used in each training
TRAINING_RATIO = 5  # the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # as per the paper

# size of image
img_rows = 56
img_cols = 56
channels = 3
picklefile_path = os.environ['XTRAIN'] # path to Xtrain array
output_dir = 'gan' # path to save outputs to

# define loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

# make generator
def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", 
    and outputs images of size 56x56x3."""
    model = Sequential()

    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())

    # 128 is the number of filters - slowly decrease
    depth = 128
    f, f2 = depth, int(depth/2)

    # 7 is the starting dimension. 7 * 8 = 56
    # (8 = 2**3, where 3 is the number of conv2dtranspose layers)
    dim = 7

    conv_window = 4 # height and width of convolution window, reduced from 5 to 4
    stride = 2
    channels = 3

    model.add(Dense(depth * dim * dim)) 
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((dim, dim, f), input_shape=(f * dim * dim,)))

    model.add(Conv2DTranspose(f, conv_window, strides=stride, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Convolution2D(f2, conv_window, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(f2, conv_window, strides=stride, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(f2, conv_window, strides=stride, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(f2, conv_window, strides=stride, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # because we normalized training inputs to lie in the range [-1, 1] by minusing and dividng by 127.5,
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.
    model.add(Convolution2D(channels, conv_window, padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, 
    representing whether the input is real or generated. """

    depth = 64 # arbitrary number of filters
    conv_window = 4 # reduced to 4
    stride = 2
    channels = 3
    model = Sequential()

    model.add(Convolution2D(depth, conv_window, padding='same', input_shape=(img_rows, img_rows, channels)))
    model.add(LeakyReLU())

    model.add(Convolution2D(depth*2, conv_window, kernel_initializer='he_normal', strides=stride)) 
    model.add(LeakyReLU())

    model.add(Convolution2D(depth*2, conv_window, kernel_initializer='he_normal', strides=stride)) 
    model.add(LeakyReLU())

    model.add(Convolution2D(depth*2, conv_window, kernel_initializer='he_normal', padding='same', strides=stride))
    model.add(LeakyReLU())
    model.add(Flatten())

    model.add(Dense(depth*16, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(channels, kernel_initializer='he_normal'))
    return model

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def generate_images(generator_model, output_dir, epoch):
    """Feeds 16 random seeds into the generator saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(16, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))

    # plot 16 images into a 4x4 square
    plt.figure(figsize=(20,20))
    filename = "%s/logo_%d.png" % (output_dir, epoch)
    for i in range(test_image_stack.shape[0]):
        plt.subplot(4, 4, i+1)
        image = test_image_stack[i, :, :, :]
        image = np.reshape(image, [img_rows, img_cols, channels])
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')


# load X_train data as an array with shape (x, height, width, channel) where x = number of images or batch size
X_train = np.load(picklefile_path)

# normalise the data
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# initialize the generator and discriminator
generator = make_generator()
discriminator = make_discriminator()

# the generator_model is used when we want to train the generator layers.
# as such, we ensure that the discriminator layers are not trainable.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

generator_input = Input(shape=(100,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss) # use Adam parameters

# now that generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(100,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])

# run these samples through the discriminator as well. 
# note that we never really use the discriminator output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# the gradient penalty loss function requires the input averaged samples to get gradients. 
# however, Keras loss functions can only have two arguments, y_true and y_pred. 
# we get around this by making a partial() of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # functions need names or Keras will throw an error


discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

# labelling data
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

# to track losses
discriminator_loss = []
generator_loss = []
loss = []

for epoch in range(10000):
    np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))

    # generate randomly rotated/shifted/flipped images from original images
    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    datagen.fit(X_train)
    minibatches_size = BATCH_SIZE * TRAINING_RATIO #steps?
    for img_batch in datagen.flow(X_train, batch_size=BATCH_SIZE):
        for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            # discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
            # discriminator_minibatches = img_batch
            for j in range(TRAINING_RATIO):
                # image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                image_batch = img_batch
                noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
                d_loss = discriminator_model.train_on_batch([image_batch, noise],
                                                                            [positive_y, negative_y, dummy_y])
                discriminator_loss.append(d_loss)
            a_loss = generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100), positive_y)
            generator_loss.append(a_loss)
            loss.append([epoch, d_loss[0], a_loss])
        break

    if epoch % 500 == 0:
        # generate images
        generate_images(generator, output_dir, epoch)

        # print loss messages
        log_mesg = "%d: [D loss: %f, acc: %f]" % (epoch, d_loss[0], d_loss[1])
        log_mesg_1 = "%s [A loss: %f]" % (log_mesg, a_loss)
        print(log_mesg_1)
        
    if epoch % 1000 == 0:
        # save discriminator model locally
        try:
            filename = 'gan_models/discr_model_' + str(epoch)
            discr_model = discriminator_model
            print('saving discriminator model locally')
            discr_model.save(filename)
        except:
            print('unable to save discriminator model locally')
            pass

        # save adversarial model locally
        try:
            filename = 'gan_models/adv_model_' + str(epoch)
            adv_model = generator_model
            print('saving adversarial model locally')
            adv_model.save(filename)
        except:
            print('unable to save adversarial model locally')
            pass

        # save losses locally
        d_loss_name = 'd_loss_' + str(epoch)
        a_loss_name = 'a_loss_' + str(epoch)
        np.save(d_loss_name, np.asarray(discriminator_loss))
        np.save(a_loss_name, np.asarray(generator_loss))
        print('saved losses locally')

        # plot losses
        epoch = []
        discr = []
        adv = []
        for loss_epoch in loss:
            epoch.append(loss_epoch[0])
            discr.append(loss_epoch[1])
            adv.append(loss_epoch[2])
            
        df = pd.DataFrame([epoch, discr, adv]).transpose()

        df.plot(x=0, y=1, figsize=(15,8))
        plt.savefig('discriminator_' + loss_name)

        df.plot(x=0, y=2, figsize=(15,8))
        plt.savefig('adversarial_' + loss_name)

# upload direct to aws s3
# bashCommand = "aws s3 cp -r gan s3://gan-project"
# import subprocess
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

# bashCommand2 = "aws s3 cp -r gan_models s3://gan-project"
# process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
# output, error = process2.communicate()
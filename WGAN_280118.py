"""
From https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
"""
import argparse
import os
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from functools import partial
import numpy as np
from PIL import Image
import pickle

BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
    of size 28x28x1."""
    model = Sequential()

    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())

    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        bn_axis = 1 # define the axis to be normalised, usually the features axis
    else:
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
        bn_axis = -1

    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.

    model.add(Convolution2D(3, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, 
    representing whether the input is real or generated. """

    model = Sequential()

    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(3, 28, 28)))
    else:
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(28, 28, 3)))
    model.add(LeakyReLU())

    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2])) 
    model.add(LeakyReLU())

    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(3, kernel_initializer='he_normal'))
    return model

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 4
    image_list = [image_stack[i, :, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images

class RandomWeightedAverage(_Merge):

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(10, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output)  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated files to")
args = parser.parse_args()

# First we load the image data, reshape it and normalize it to the range [-1, 1]

# load X_train data as an array with shape (x, height, width, channel) where x = number of images or batch size
picklefile_path = 'X_train_28_1503'
X_train = np.load(picklefile_path) * 255.0 # as it was pre-processed

X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Now we initialize the generator and discriminator.
generator = make_generator()
discriminator = make_discriminator()

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(100,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
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

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])

# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error


discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

for epoch in range(10000):
    np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                         [positive_y, negative_y, dummy_y]))
        generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100), positive_y))
    # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
    if epoch % 500 == 0:
        generate_images(generator, args.output_dir, epoch)

        # save discriminator model locally
        try:
            filename = 'discr_model_' + str(epoch)
            discr_model = discriminator_model
            print('saving discriminator model locally')
            discr_model.save(filename)
        except:
            print('unable to save discriminator model locally')
            pass

        # save adversarial model locally
        try:
            filename = 'adv_model_' + str(epoch)
            adv_model = generator_model
            print('saving adversarial model locally')
            adv_model.save(filename)
        except:
            print('unable to save adversarial model locally')
            pass

        # save loss plots locally
        try:
            filename = 'discriminator_loss_' + str(epoch) + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(discriminator_loss, f)
            print('saving discr loss locally')
        except:
            print('unable to save discriminator loss locally')
            pass

        # save accuracy plots locally
        try:
            filename = 'generator_loss_' + str(epoch) + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(generator_loss, f)
            print('saving generator loss locally')
        except:
            print('unable to save generator loss locally')
            pass

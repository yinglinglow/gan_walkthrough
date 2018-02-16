"""
Base code from: 
https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py

Optimisation done for:
56 x 56 RGB images

Import pre-processed image array saved in S3 bucket

To run:
mkdir dcgan
mkdir dcgan_models
export XTRAIN=X_train_56_1700.pkl
export CODE=DCGAN_150218_11am
export DATE=150218
aws s3 cp s3://gan_project/$XTRAIN .
tmux
python3 $CODE.py

"""

# necessary when running on AWS EC2
import matplotlib
matplotlib.use('Agg')

# import requirements
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import io
import random
import pandas as pd
from PIL import Image, ImageOps
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

# fix random seed for reproducibility
np.random.seed(0) 

# define DCGAN class
class DCGAN(object):
    def __init__(self, img_rows=56, img_cols=56, channel=3): # RGB = 3 channels. B&W = 1 channel
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        
    def discriminator(self):
        if self.D: # if self.D exists already just use it, otherwise create
            return self.D
        
        self.D = Sequential()
        depth = 64 # number of filters (for identifying different features in the image), arbitrary
        conv_window = 6 # height and width of convolution window. decreased from 5 to 4
        stride = 2 # number of pixels the window slides across
        dropout = 0.4 # amount to dropout to prevent overfitting
        
        input_shape = (self.img_rows, self.img_cols, self.channel) # shape of input images
        self.D.add(Conv2D(depth*1, conv_window, strides=stride, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2)) # allows for small gradient when unit is not active
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, conv_window, strides=stride, padding='same')) # increase number of filters - arbitrary
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, conv_window, strides=stride, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, conv_window, strides=stride, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability whether image is real or fake
        self.D.add(Flatten())
        self.D.add(Dense(1)) # output 1 node
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D
    
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64 * 4 # number of filters, arbitrary
        f2, f4, f8 = int(depth/2), int(depth/4), int(depth/8) # number of output filters in the convolution
        # values to slowly upscale the image
        # int() truncates decimal points towards zero

        conv_window = 6 # height and width of convolution window. reduce from 5 to 4
        
        input_dim = 100
        # 100-dimensional noise (uniform distribution between -1.0 to 1.0)

        dim = 7 # final desired output shape (56) divided by 8 (2**3 = number of UpSampling2D layers. alternatively can use fractional strides too)
        self.G.add(Dense(dim * dim * depth, input_dim=input_dim ))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu')) # use relu for all layers as per DCGAN guidelines
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(f2, conv_window, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(f4, conv_window, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(f4, conv_window, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(f8, conv_window, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 56 x 56 x 3 channel image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(self.channel, conv_window, padding='same'))
        self.G.add(Activation('sigmoid')) # changed back to sigmoid - 0 to 1 range
        self.G.summary()
        return self.G
    
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy']) # to compile model - train discriminator on its own.
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8) # next time try with adam?
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator()) # generator gives fake images to discriminator (which also receives real images), discriminator classifies.
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy']) # change loss function to mse because of change to tanh
        return self.AM

class LOGO_DCGAN(object):
    def __init__(self):
        self.img_rows = 56
        self.img_cols = 56
        self.channel = 3
        self.x_train = X_train

        self.DCGAN = DCGAN()
        
        # try to load, else create new models
        try:
            self.D = load_model('~/gan_project/discr_model_6999')
            self.G = load_model('~/gan_project/discr_model_6999')
            self.AM = load_model('~/gan_project/adv_model_6999')
            self.DM = load_model('~/gan_project/discr_model_6999')
            print('loaded models')
        except:
            self.D = None
            self.G = None 
            self.AM = None  
            self.DM = None
            print('created new models')

        self.generator = self.DCGAN.generator()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
            
        # generate randomly rotated/flipped images from original training images
        datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
        datagen.fit(X_train)

        # generate lists for loss and accuracy
        loss = []
        acc = []

        for img_batch in datagen.flow(X_train, batch_size=batch_size):
            for i in range(train_steps):
                
                # generate one batch of rotated training images
                images_train = img_batch

                # generate one batch of fake images from noise
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                images_fake = self.generator.predict(noise)

                # join them together
                x = np.concatenate((images_train, images_fake))

                # generate labels
                y = np.ones([2*batch_size, 1]) # label training images as 1
                y[batch_size:, :] = 0 # label fake images as 0

                # train discriminator
                d_loss = self.discriminator.train_on_batch(x, y)

                # train adversarial
                y = np.ones([batch_size, 1])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                a_loss = self.adversarial.train_on_batch(noise, y)
                
                # return log messages
                log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                print(log_mesg)

                # append loss and acc
                loss.append(list((i, d_loss[0], a_loss[0])))
                acc.append(list((i, d_loss[1], a_loss[1])))

                if save_interval>0:
                    if (i+1)%save_interval==0:
                        # plot images
                        self.plot_images(save2file=True, samples=noise_input.shape[0],\
                            noise=noise_input, step=(i+1))
                        
                        # save discriminator model locally
                        try:
                            filename = 'dcgan_models/discr_model_' + str(i) + date
                            discr_model = self.discriminator
                            print('saving discriminator model locally')
                            discr_model.save(filename)
                        except:
                            print('unable to save discriminator model locally')
                            pass

                        # save adversarial model locally
                        try:
                            filename = 'dcgan_models/adv_model_' + str(i) + date
                            adv_model = self.adversarial
                            print('saving adversarial model locally')
                            adv_model.save(filename)
                        except:
                            print('unable to save adversarial model locally')
                            pass

                        # save generator locally
                        try:
                            filename = 'dcgan_models/gen_' + str(i) + date
                            gen = self.generator
                            print('saving generator locally')
                            gen.save(filename)
                        except:
                            print('unable to save generator locally')
                            pass
                        
                        # save losses and accuracy locally
                        loss_name = 'loss_' + str(i)
                        acc_name = 'acc_' + str(i)
                        np.save(loss_name, np.asarray(loss))
                        np.save(acc_name, np.asarray(acc))
                        print('saved losses and accuracy locally')

            break # otherwise the trained images will generate (rotate, flip) infinitely

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'dcgan/logo.png'
        if fake: # return fake images
            if noise is None: # no training done
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "dcgan/logo_%d.png" % step
            images = self.generator.predict(noise)
        else: # return the training images
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(20,20))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename) # save fake images
            plt.close('all')
        else:
            plt.show()
        
if __name__ == '__main__':

    # load X_train data as an array with shape (x, height, width, channel) where x = number of images or batch size
    picklefile_path = os.environ['XTRAIN']
    date = os.environ['DATE']
    X_train = np.load(picklefile_path).astype(np.float32)/255.0
    
    # instantiate GAN model
    logo_dcgan = LOGO_DCGAN()

    # start training
    logo_dcgan.train(train_steps=10000, batch_size=256, save_interval=1000) # runs for 10000 epochs, saves every 1000
    logo_dcgan.plot_images(fake=True)
    logo_dcgan.plot_images(fake=False, save2file=True)

    # write files to aws s3 once done
    import subprocess
    bashCommand = "aws s3 cp -r dcgan s3://dcgan"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand2 = "aws s3 cp -r dcgan_models s3://dcgan"
    process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
    output, error = process2.communicate()
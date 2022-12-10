import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses

class AlexNet:
    def __init__(self, dataset):
        pass

    def build(self):
        model = models.Sequential()

        # 1st
        model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
        #model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((3, 3), strides=2))

        model.add(256, 5, strides = 1, padding='same')
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(()))
        

        model.add()
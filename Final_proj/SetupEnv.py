import numpy as np
import struct
from array import array
class Setup:
    def __init__(self, training_img, training_label, test_img, test_label):
        self.training_img = training_img
        self.training_label = training_label
        self.test_img = test_img
        self.test_label = test_label

    def read_img_label(self, img_path, label_path):
        labels = []
        with open(label_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(img_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            img_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0]*rows*cols)
        for i in range(size):
            img = np.array(img_data[i*rows*cols:(i+1)*rows*cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_img_label(self.training_img, self.training_label)
        x_test, y_test = self.read_img_label(self.test_img, self.test_label)
        return (x_train, y_train), (x_test, y_test)

    
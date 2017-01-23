import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
import numpy as np
from random import sample, randint, uniform
# from sklearn.model_selection import train_test_split

WIDTH = 64
HEIGHT = 64
CHANNEL = 3

def load_img(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pick_image_filename(x, steering):
    i = randint(0, 2)
    if i == 1:
        steering1 = steering + 0.25
    elif i == 2:
        steering1 = steering - 0.25
    else:
        steering1 = steering
    return x[i], steering1

def trans_image(image, steering):
    return image, steering

def flip_image(image, steering):
    if randint(0, 1) == 1:
        return cv2.flip(image, 1), -steering
    return image, steering

def argument_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * uniform(0.3, 1.3)


def generator(batch_size, data):
    images = np.zeros((batch_size, WIDTH, HEIGHT, CHANNEL))
    steerings = np.zeros(batch_size)
    while True:
        samples = sample(data, batch_size)
        for i in range(batch_size):
            x = samples[i]
            steering = x[3]
            filename, steering = pick_image_filename(x, steering)
            image = load_img(filename)
            image, steering = trans_image(image, steering)
            image, steering = flip_image(image, steering)
            image = argument_brightness(image)
            image = crop(image)
            image = resize(image)
            images[i] = image
            steerings[i] = steering
        yield images, steerings

def build_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Flatten())
    # model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.summary()
    return model

def save_model(model):
    '''save model and weights'''
    with open('model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('model.h5')

log_records = []

with open("driving_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        log_records.append(row)

print("length of log is:", len(log_records))

# normalize
# images = np.array(images).astype('float32')
# images = images / 127.5 - 1.
# # images = np.array(images)
#
#
# # size of image
# shape = images[0].shape
#
# # train
#
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['mean_squared_error'])
#
# for i in range(10):
#     # training set and validation set
#     train_images, val_images, train_angles, val_angles = train_test_split(images, angles, test_size=0.2)
#     history = model.fit(train_images, train_angles,
#                         batch_size=128, nb_epoch=2,
#                         verbose=1, validation_data=(val_images, val_angles))
#

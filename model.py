import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten, ELU, Lambda
import numpy as np
from random import sample, randint, uniform
from sklearn.model_selection import train_test_split

WIDTH = 64
HEIGHT = 64
CHANNEL = 3

def load_img(filename):
    # print(filename)
    img = cv2.imread(filename)
    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img)
    return img

def pick_image_filename(x, steering):
    i = randint(0, 2)
    if i == 1:
        steering1 = steering + 0.25
    elif i == 2:
        steering1 = steering - 0.25
    else:
        steering1 = steering
    return x[i].strip(), steering1

def trans_image(image, steering):
    rows, cols, _ = image.shape
    trans_x = randint(-50, 51)
    trans_steering = steering + trans_x * 0.004
    trans_y = randint(-10, 11)
    trans_M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    trans_image = cv2.warpAffine(image, trans_M, (cols, rows))
    return trans_image, trans_steering

def flip_image(image, steering):
    if randint(0, 1) == 1:
        return cv2.flip(image, 1), -steering
    return image, steering

def argument_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * uniform(0.3, 1.3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def crop(image):
    return image[30:140, :]

def resize(image):
    return cv2.resize(image,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)

def generator(batch_size, data):
    images = np.zeros((batch_size, WIDTH, HEIGHT, CHANNEL))
    steerings = np.zeros(batch_size)
    while True:
        samples = sample(data, batch_size)
        for i in range(batch_size):
            x = samples[i]
            # print(x)
            steering = float(x[3])
            filename, steering = pick_image_filename(x, steering)
            # print(filename, steering)
            image = load_img(filename)
            # print(image)
            image, steering = trans_image(image, steering)
            # print(image, steering)
            image, steering = flip_image(image, steering)
            # print(image, steering)
            image = argument_brightness(image)
            # print(image)
            image = crop(image)
            # print(image)
            image = resize(image)
            # print(image)
            images[i] = image
            steerings[i] = steering
        yield images, steerings

def val_generate(data):
    size = len(data)
    images = np.zeros((size, WIDTH, HEIGHT, CHANNEL))
    steerings = np.zeros(batch_size)
    for i in range(size):
        x = data[i]
        steering = float(x[3])
        filename = x[0].strip()
        image = load_img(filename)
        image = crop(image)
        image = resize(image)
        images[i] = image
        steerings[i] = steering
    return images, steerings

def build_model():
    row, col, ch = HEIGHT, WIDTH, CHANNEL
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
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
    reader = csv.reader(f)
    for row in reader:
        log_records.append(row)
print("length of log is:", len(log_records))
training_set, validation_set = train_test_split(log_records, test_size=0.1, random_state=42)
model = build_model()
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(generator(5000, training_set), 5000, nb_epoch=100, verbose=1,
                              validation_data=val_generate(validation_set), nb_val_samples=1000)
save_model(model)

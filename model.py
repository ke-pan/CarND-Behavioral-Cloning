import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Convolution2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split

# prepare features and labels
images = []
angles = []
with open("driving_log.csv") as f:
    log_csv = csv.reader(f)
    for row in log_csv:
        images.append(cv2.imread(row[0]))
        angles.append(row[3])

# training set and validation set
train_images, val_images, train_angles, val_angles = train_test_split(images, angles, test_size=0.33)

# size of image
shape = train_images[0].shape

# normalize
train_images = np.array(train_images).astype('float32')
val_images = np.array(val_images).astype('float32')
train_images = train_images / 255 - 0.5
val_images = val_images / 255 - 0.5

# architecture
model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

# train

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_images, train_angles,
                    batch_size=128, nb_epoch=5,
                    verbose=1, validation_data=(val_images, val_angles))

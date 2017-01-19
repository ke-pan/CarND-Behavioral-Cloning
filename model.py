import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split

# prepare features and labels
images = []
angles = []
with open("driving_log.csv") as f:
    log_csv = csv.reader(f)
    for row in log_csv:
        img = cv2.imread(row[0])
        images.append(cv2.resize(img[::-1], (160, 80))
        angles.append(row[3])

# normalize
images = np.array(images).astype('float32')
images = images / 255 - 0.5
# images = np.array(images)


# size of image
shape = images[0].shape

# architecture
model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), border_mode='same'))
model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# train

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

for i in range(10):
    # training set and validation set
    train_images, val_images, train_angles, val_angles = train_test_split(images, angles, test_size=0.2)
    history = model.fit(train_images, train_angles,
                        batch_size=128, nb_epoch=2,
                        verbose=1, validation_data=(val_images, val_angles))

# save model and weights
with open('model.json', 'w') as f:
    f.write(model.to_json())

model.save_weights('model.h5')

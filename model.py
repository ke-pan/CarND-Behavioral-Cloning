import csv
import cv2
from PIL import Image
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
        # img = cv2.imread(row[0])[::-1]
        img = Image.open(row[0])
        img = img.resize((160,80))
        img = np.asarray(img)
        images.append(img)
        angles.append(row[3])

# normalize
images = np.array(images).astype('float32')
images = images / 127.5 - 1.
# images = np.array(images)


# size of image
shape = images[0].shape

# architecture
model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), border_mode='same'))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Flatten())
# model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# train

model.compile(loss='mse', optimizer='adam')
              

# training set and validation set
train_images, val_images, train_angles, val_angles = train_test_split(images, angles, test_size=0.2)
history = model.fit(train_images, train_angles,
                    batch_size=256, nb_epoch=25,
                    verbose=1, validation_data=(val_images, val_angles))

# save model and weights
with open('model.json', 'w') as f:
    f.write(model.to_json())

model.save_weights('model.h5')

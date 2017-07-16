import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images = []
augmented_measurements = []

for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(cv2.flip(image, 1))
    augmented_images.append(image)
    augmented_measurements.append(measurement*-1.1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

shape = (160,320,3)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape))
model.add(Convolution2D(6,5,5, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))
#model.add(Dropout(0.25))
model.add(Convolution2D(6,5,5, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=5, shuffle=True)

model.save('model.h5')


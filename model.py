import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

images = []
measurements = []

correction = 0.4 # this is a parameter to tune

def get_image(path):
    filename = path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    return cv2.imread(current_path)

for line in lines:
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    img_center = get_image(line[0])
    img_left = get_image(line[0])
    img_right = get_image(line[0])

    images.extend([img_center, img_left, img_right])
    measurements.extend([steering_center, steering_left, steering_right])
    
augmented_images = []
augmented_measurements = []

for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

shape = (160,320,3)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5), input_shape=shape)
model.add(Cropping2D(cropping=((50,20), (0,0))))
# NVidia
model.add(Convolution2d(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2d(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2d(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2d(64,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2d(64,3,3, subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# LeNet
# model.add(Convolution2D(6,5,5, activation='relu', border_mode='valid'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))
# model.add(Dropout(0.25))
# model.add(Convolution2D(6,5,5, activation='relu', border_mode='valid'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=3, shuffle=True)

model.save('model.h5')


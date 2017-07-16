import argparse
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from pathlib import Path
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

weights_file = 'model.h5'

parser = argparse.ArgumentParser(description='Driver Training')
parser.add_argument(
    'data_folder',
    type=str,
    help='Path to data.'
)
parser.add_argument(
    'number_epochs',
    type=int,
    default='5',
    help='Number of epochs.'
)
args = parser.parse_args()

samples = []
with open(args.data_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

correction = 0.4 # this is a parameter to tune
num_transformations = 6

def get_image(path):
    filename = path.split('/')[-1]
    current_path = args.data_folder + '/IMG/' + filename
    return cv2.imread(current_path)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                img_center = get_image(batch_sample[0])
                img_left = get_image(batch_sample[1])
                img_right = get_image(batch_sample[2])

                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])

            augmented_images = []
            augmented_measurements = []

            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

shape = (160,320,3)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape))
model.add(Cropping2D(cropping=((50,20), (0,0))))
# NVidia
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

if Path(weights_file).exists():
    model.load_weights(weights_file)

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

# model.fit(X_train, y_train, validation_split=0.2, nb_epoch=5, shuffle=True)
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples * num_transformations),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples * num_transformations),
                    nb_epoch=args.number_epochs)

model.save(weights_file)
# Chandru, Jeyanth
# 1001-359-339
# 2017-11-27
# Assignment_06_01

import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('tf')

def load_images_from_folder(folder):
    images = []
    class_img = []
    for i, filename in enumerate(os.listdir(folder)):
        if filename.endswith(".DS_Store"):
            continue
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                if len(class_img) == 0:
                    class_img = [i]
                else:
                    class_img.append(i)
        else:
            for file in os.listdir(os.path.join(folder, filename)):
                if file.endswith(".jpg"):
                    img = cv2.imread(os.path.join(folder,filename, file))
                    if img is not None:
                        images.append(img)
                        if len(class_img) == 0:
                            class_img = [i]
                        else:
                            class_img.append(i)
    return (images, class_img)
train_images, train_class = load_images_from_folder("Train")
train_images = np.array(train_images)/255.0
train_class = np.array(train_class)
y_train = np_utils.to_categorical(train_class)
test_folder_location = input("Enter the Full path of the test folder: ")
test_images, test_class = load_images_from_folder(test_folder_location)
test_images = np.array(test_images)/255.0
test_class = np.array(test_class)
y_test = np_utils.to_categorical(test_class)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

epochs = 65
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#adam = Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# checkpoint
# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
print(model.summary())
model.fit(train_images, y_train, validation_data=(test_images, y_test), epochs=epochs, batch_size=32, verbose=1)
# Final evaluation of the model
scores = model.evaluate(test_images, y_test, verbose=0)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
print("Accuracy: %.2f%%" % (scores[1]*100))

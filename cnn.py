from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

'''
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])'''


import keras
'''
model = Sequential()
model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/rahul/Downloads/dlcvnlp/cnn/dogcat_new/familyimage',
                                                 target_size = (64, 64),
                                                 batch_size = 32)
test_set = test_datagen.flow_from_directory('/home/rahul/Downloads/dlcvnlp/cnn/dogcat_new/familyimage',
                                            target_size = (64, 64),
                                            batch_size = 32)
mod = model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 100)
training_set.class_indices
model.save("familylenet1.h5")
print("Saved model to disk")'''




#Part 3 - Making new predictions

from keras.models import load_model
model=load_model('familylenet1.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('r.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'anna'
    print(prediction)
if result[0][1]==1:
    prediction = 'lion'
    print(prediction)
if result[0][2]==1:
    prediction = 'rahul'
    print(prediction)
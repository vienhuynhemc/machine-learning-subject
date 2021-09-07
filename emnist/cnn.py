def get_images(img_file, number):
    f = open(img_file, "rb") # Open file in binary mode
    f.read(16) # Skip 16 bytes header
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    return images

def get_labels(label_file, number):
    l = open(label_file, "rb") # Open file in binary mode
    l.read(8) # Skip 8 bytes header
    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
    return labels



import numpy as np

from mnist import MNIST

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from keras.utils import np_utils

mndata = MNIST('data')
#This will load the train and test data

X_train, y_train = mndata.load('data/emnist-byclass-train-images-idx3-ubyte',
                               'data/emnist-byclass-train-labels-idx1-ubyte')
X_test, y_test = mndata.load('data/emnist-byclass-test-images-idx3-ubyte',
                             'data/emnist-byclass-test-labels-idx1-ubyte')



# Train
print("Train set")

TRAINING_SIZE = 100000
TEST_SIZE = 40000
n_classes = 10
# ---------------------------------
X_train = get_images("data/emnist-byclass-train-images-idx3-ubyte",TRAINING_SIZE)

X_train = np.array(X_train) / 255.0
X_train = X_train.astype('float32')

y_train = get_labels("data/emnist-byclass-train-labels-idx1-ubyte",TRAINING_SIZE)
# -----------------------------------

X_test = get_images("data/emnist-byclass-test-images-idx3-ubyte",TEST_SIZE)

X_test = np.array(X_test) / 255.0
X_test = X_test.astype('float32')

y_test = get_labels("data/emnist-byclass-test-labels-idx1-ubyte",TEST_SIZE)

y_train = np.array(y_train) / 255.0
y_test = np.array(y_test) / 255.0

print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train,62)
Y_test = np_utils.to_categorical(y_test,62)
print("Shape after one-hot encoding: ", Y_train.shape)



# buiding a linear stack of layers with the sequencial model
model = Sequential()
# hidden layer
model.add(Dense(100, input_shape=(784,), activation='relu'))
# output layer
model.add(Dense(62,activation='softmax'))


# looking at the model summary
model.summary()

# compiling the sequetial model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training model for 10 epochs
model.fit(X_train,Y_train,batch_size=128,epochs=5,validation_data=(X_test,Y_test))



print("Test set")




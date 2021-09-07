### We begin my importing the data from the data folder.

# %%

import tensorflow as tf

from fg import freeze_graph

import numpy as np
from mnist import MNIST


mndata = MNIST('data')
# This will load the train and test data
X_train, y_train = mndata.load('data/emnist-byclass-train-images-idx3-ubyte',
                               'data/emnist-byclass-train-labels-idx1-ubyte')
X_test, y_test = mndata.load('data/emnist-byclass-test-images-idx3-ubyte',
                             'data/emnist-byclass-test-labels-idx1-ubyte')

# Convert data to numpy arrays and normalize images to the interval [0, 1]
X_train = np.array(X_train) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)




# %% md

# Getting Data ready for pre-processing

# %%

# Reshaping all images into 28*28 for pre-processing
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

# %%

from matplotlib import pyplot as plt

# Display a random image
plt.imshow(X_train[0])
plt.show



# %%

# Y'all can see how an image array looks like. all float values b/w 0 and 1
m = X_train[2]
print(m)


print("ABC")

# # %% md
#
# ### Now we perform Image preprocessing. We reverse and rotate all train and test images
#
# # %%
#
# # for train data
# for t in range(697932):
#     X_train[t] = np.transpose(X_train[t])
#
# # checking
# plt.imshow(X_train[0])
# plt.show
#
# # for test data
# for t in range(116323):
#     X_test[t] = np.transpose(X_test[t])
#
# # checking
# plt.imshow(X_test[1])
# plt.show
#
# print('Process Complete: Rotated and reversed test and train images!')
#
# # %%
#
# # Checking the last train image, just to be sure!
# m = X_train[697931]
# plt.imshow(m)
# plt.show()
#
# # %%
#
#
# # %% md
#
# ### Reshaping train and test data again for input into model
#
# # %%
#
# X_train = X_train.reshape(X_train.shape[0], 784, 1)
# X_test = X_test.reshape(X_test.shape[0], 784, 1)
#
# # %% md
#
# ### Creation of model
#
# # %%
#
# from keras.models import Sequential
# from keras import optimizers
# from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM
# from keras import backend as K
# from keras.constraints import maxnorm
#
#
# def resh(ipar):
#     opar = []
#     for image in ipar:
#         opar.append(image.reshape(-1))
#     return np.asarray(opar)
#
#
# from keras.utils import np_utils
#
# train_images = X_train.astype('float32')
# test_images = X_test.astype('float32')
#
# train_images = resh(train_images)
# test_images = resh(test_images)
#
# train_labels = np_utils.to_categorical(y_train, 62)
# test_labels = np_utils.to_categorical(y_test, 62)
#
# K.set_learning_phase(1)
#
# model = Sequential()
#
# model.add(Reshape((28, 28, 1), input_shape=(784,)));
#
# # add the layer below for an accuracy of 89%.(Training time - over 20 hours)
# # model.add(Convolution2D(32, (5,5), input_shape=(28,28,1),
# # activation='relu',padding='same', kernel_constraint=));
# model.add(Convolution2D(32, (5, 5), activation='relu'));
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
#
# # model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)));
#
# model.add(Dropout(0.5));
#
# model.add(Dense(62, activation='softmax'));
#
# # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# # opt = optimizers.Adadelta()
# opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0);
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']);
#
# # %%
#
#
# # %% md
#
# ### Training of model and evaluation
#
# # %%
#
# print(model.summary());
# history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=128, epochs=20);
#
# # %%
#
# # evaluating model on test data. will take time
# scores = model.evaluate(test_images, test_labels, verbose=0);
# print("Accuracy: %.2f%%" % (scores[1] * 100));
#
# # %% md
#
# ## Creating model history graphs
#
# # %%
#
# print(history.history.keys());
# # summarize history for accuracy
# plt.plot(history.history['acc']);
# plt.plot(history.history['val_acc']);
# plt.title('Model Accuracy');
# plt.ylabel('Accuracy');
# plt.xlabel('Epoch');
# plt.legend(['Train', 'Test'], loc='upper left');
# plt.grid();
# plt.show();
# # summarize history for loss
# plt.plot(history.history['loss']);
# plt.plot(history.history['val_loss']);
# plt.title('Model loss');
# plt.ylabel('Loss');
# plt.xlabel('Epoch');
# plt.legend(['Train', 'Test'], loc='upper left');
# plt.grid();
# plt.show();
#
# # %%
#
# objects = ('RMSDrop', 'Adam', 'Adamax', 'SGD', 'Adadelta');
# y_pos = np.arange(len(objects));
# performance = [86.2, 85.39, 89.53, 84.29, 87.11];
#
# plt.bar(y_pos, performance, align='center', alpha=0.5);
# plt.xticks(y_pos, objects);
# plt.ylabel('Accuracy');
# plt.title('Optimizers');
# plt.ylim(50, 100);
# plt.show();
#
# # %% md
#
#
# # %% md
#
# # Freezing the graph for android Import
#
# # %%
#
# frozen_graph = freeze_graph(K.get_session(), output_names=[model.output.op.name]);
# tf.train.write_graph(frozen_graph, '.', 'C:/Users/84168/PycharmProjects/amnist/PBfile8953.pb', as_text=False);
# print(model.input.op.name);
# print(model.output.op.name);
#
# # %% md
#
# ## Predicting a single image using the model
#
# # %%
#
# m = X_test[258].reshape(28, 28);
# plt.imshow(m);
# plt.show;
# print('prediction: ' + str(model.predict_classes(X_test[258].reshape(1, 784))));
#
# # %% md
#
# ## Saving the model
#
# # %%
#
# from keras.models import load_model
# from keras.models import model_from_json
#
# model_json = model.to_json();
# with open("model.json", "w") as json_file:
#     json_file.write(model_json);
# # saves the model info as json file
#
# model.save_weights("model.h5")
# # Creates a HDF5 file 'model.h5'
#
# # %% md
#
# # For usage of this model to predict words, open segment.ipynb
#
# # %%
#
#
# # %%
#
#
# # %%
#


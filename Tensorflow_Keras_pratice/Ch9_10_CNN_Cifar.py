import matplotlib.pyplot as plt
import numpy as np
import os
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import load_model

np.random.seed(10)


(train_data, train_label), (test_data, test_label) =cifar10.load_data()
train_data_4D = train_data.reshape(
    train_data.shape[0], 32, 32, 3).astype('float')
test_data_data_4D = test_data.reshape(
    test_data.shape[0], 32, 32, 3).astype('float')

train_data_4D_normalized = train_data_4D / 255
test_data_data_4D_normalized = test_data_data_4D / 255

train_label_1hot = np_utils.to_categorical(train_label)
test_label_1hot = np_utils.to_categorical(test_label)


def build_cnn():
    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        input_shape=(32, 32, 3),
        activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


model = build_cnn()

dir = 'Tensorflow_Keras_pratice_models'
file_name = 'Ch9_10_CNN_Cifar_model_weights.h5'
save_path = os.path.join(dir, file_name)
try:
    model = load_model(save_path)
    print('Start with a exist model.')
except:
    print('Start with a new model.')

train_history = model.fit(
    x=train_data_4D_normalized,
    y=train_label_1hot,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=2)

model.save(save_path)

def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('History')
    plt.xlabel('Epoch')
    plt.ylabel('train')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'acc', 'val_acc')
score = model.evaluate(test_data_data_4D_normalized, test_label_1hot)
print(score[1])
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt
import pandas as pd

(train_data, train_label_1hot), (test_data, test_label_1hot) = mnist.load_data()


def data_preprocess(train_data, train_label, test_data, test_label):
    # reshape 2d to 1d
    processed_train_data = train_data.reshape(60000, 784).astype('float32')
    processed_test_data = test_data.reshape(10000, 784).astype('float32')

    # normalization
    processed_train_data /= 255
    processed_test_data /= 255

    # one-hot-encoding
    train_label_1hot = np_utils.to_categorical(train_label)
    test_label_1hot = np_utils.to_categorical(test_label)
    return processed_train_data, train_label_1hot, processed_test_data, test_label_1hot


(train_data, train_label), (test_data, test_label) = mnist.load_data()
train_data, train_label_1hot, test_data, test_label_1hot = data_preprocess(
    train_data, train_label, test_data, test_label)


def build_mlp_model(activation='relu'):
    model = Sequential()
    model.add(Dense(units=1024, input_dim=784,
                    kernel_initializer='normal', activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(units=1024, input_dim=784,
                    kernel_initializer='normal', activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


model = build_mlp_model()
history = model.fit(x=train_data, y=train_label_1hot,
                    validation_split=0.2, epochs=30, batch_size=200, verbose=2)
# model = build_mlp_model(activation='sigmoid')
# history_sigmoid = model.fit(x=train_data, y=train_label,
#                     validation_split=0.2, epochs=30, batch_size=200, verbose=2)

prediction = model.predict_classes(test_data)

# def compare_2_history(history1, history2):
#     plt.plot(history1.history['acc'])
#     plt.plot(history2.history['acc'])
#     plt.title('History')
#     plt.xlabel('Epoch')
#     plt.ylabel('train')
#     plt.legend(['history_relu', 'history_sigmoid'], loc='upper left')

# compare_2_history(history, history_sigmoid)
def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('History')
    plt.xlabel('Epoch')
    plt.ylabel('train')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(history, 'acc', 'val_acc')

df = pd.DataFrame({'label':test_label, 'predict': prediction})
print(df)
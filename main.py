import tensorflow as tf
from tensorflow.keras.layers import Dense
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime

N_EXAMPLES = 40000

def creat_model():
    model = tf.keras.models.Sequential()
    model.add(Dense(10, activation='sigmoid', input_shape=(2,)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='rmsprop',
                loss='mean_squared_error',
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            epochs=20, batch_size=128,
            validation_data=(x_test, y_test))

    history_dict = history.history
    print(history_dict.keys())
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']

    epochs = range(1, len(acc_values) + 1)

    plt.clf()

    val_acc_values = history_dict['val_accuracy']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def creat_data(nsamples):
    examples = []
    labels = []
    for n in range(0, nsamples):
        n1 = random.randint(1, 9)
        n2 = random.randint(1, 9)

        examples.append((n1, n2))
        labels.append(n1 + n2/n1)
    return examples, labels
        
def split_data(examples, labels):
    n = len(examples)
    examples = np.asarray(examples).astype('float32')
    labels = np.asarray(labels).astype('float32')

    x_train = examples[:int(n*0.7)]
    x_test = examples[int(n*0.7):]

    y_train = labels[:int(n*0.7)]
    y_test = labels[int(n*0.7):]

    return x_train, y_train, x_test, y_test


def main(N_EXAMPLES):
    data = creat_data(N_EXAMPLES)
    x_train, y_train, x_test, y_test = split_data(data[0], data[1])
    model = creat_model()
    train_model(model, x_train, y_train, x_test, y_test)

    while True:
        value = input('Digite um valor para testar (x y): ')
        value = value.split(' ')
        value = [[int(value[0]), int(value[1])]]
        test = np.asarray(value).astype('float32')
        if test.any() == 0:
            break
        print(model.predict(test), '|', test[0][0])
    value = [2, 1]
    value = np.asarray([value]).astype('float32')
    print(value, value.shape)
    print(model.predict(value))



main(N_EXAMPLES)


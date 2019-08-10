from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt 
import numpy as np

def create_model(init):
    model = Sequential()
    model.add(Dense(100, input_shape = [28*28, ], init = init, activation = 'tanh'))
    model.add(Dense(100, init = init, activation = 'tanh'))
    model.add(Dense(100, init = init, activation = 'tanh'))
    model.add(Dense(100, init = init, activation = 'tanh'))
    model.add(Dense(10, init = init, activation = 'softmax'))
    return model


def _main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    X_train = x_train.reshape([-1,28*28]) / 255
    X_test = x_test.reshape([-1,28*28]) / 255
    #uniform initialization
    uniform_model = create_model('uniform')
    uniform_model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
    uniform_history = uniform_model.fit(x = X_train,y = Y_train,batch_size = 64,
     nb_epoch = 30, verbose = 1 , validation_data = (X_test, Y_test))
    #glorot initialization
    glorot_model = create_model('glorot_normal')
    glorot_model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
    glorot_history = glorot_model.fit(x = X_train,y = Y_train,batch_size = 64,
     nb_epoch = 30, verbose = 1 , validation_data = (X_test, Y_test))
    #visualization of accuracy
    plt.plot(uniform_history.history['val_acc'])
    plt.plot(glorot_history.history['val_acc'], '--')
    plt.legend(['uniform' , 'glorot'])
    plt.yticks(np.arange(0,1,0.1))
    plt.xticks(np.arange(0,31,2))
    plt.xlim(0,30)
    plt.ylim(0,1)
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    _main()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D ,MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X= X/255.0

CATAGORIES = ["alpha","beta","sigma","pi"]

model = Sequential()

model.add(Conv2D(64, (2,2) , input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4,4)))


model.add(Conv2D(64, (2,2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4,4)))


model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(128))
model.add(Activation("relu"))


model.add(Dense(128))
model.add(Activation("relu"))


model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

value = input("Please enter the number of times to train:\n")
value = int(value)

model.fit(X, y, batch_size=32, epochs=value, validation_split=0.1)

def accur():
    history = model.fit(X, y, batch_size=32, epochs=value, validation_split=0.1)

    test_loss, test_acc = model.evaluate(X,  y, verbose=2)

    print('\nTest accuracy:', test_acc ,"\n")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(value)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

predictions=model.predict([X])

value = input("Please enter an integer:\n")
value = int(value)

while value!=-1:
    try :
        print(CATAGORIES[int(np.argmax(predictions[value]))])
        plt.imshow(X[value], cmap=plt.cm.binary)
        plt.show()

        value = input("Please enter an integer: - (-1 to exit)\n")
        value = int(value)
    except Exception as e:
        pass

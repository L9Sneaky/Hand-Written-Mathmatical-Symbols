import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D ,MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X= X/255.0

CATAGORIES = ["alpha","beta","sigma","pi"]

model = Sequential()

model.add(Conv2D(32, (3,3) , input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation("relu"))
#model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

os.system('cls')
value = input("Please enter the number of times to train:\n>")
value = int(value)

model.fit(X, y, batch_size=512, epochs=value, validation_split=0.1,learning_rate=1e-3,dropout_rate=0.2)
os.system('cls')

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


#bruh = 'Untitled.jpg'
#bruh2 = cv2.imread(bruh)
#bruh2 = bruh2/255.0
#bruh2 = np.expand_dims(bruh2, axis=0)
#predictions=model.predict(bruh2)
#print(np.argmax(bruh2))
#print(CATAGORIES[int(np.argmax(cv2.imread(bruh)))])

hehe = "Please enter an integer: - (0 to stats \ -1 to exit)\n>"
model.evaluate(X, y, verbose=2)
value = input()
value = int(value)
while value!=-1:
    os.system('cls')
    try :
        if value == 0:
            model.summary()
            value = input(hehe)
            value = int(value)
        else:
            bruh = 'Untitled.jpg'
            bruh2 = cv2.imread(bruh)
            bruh2 = bruh2/255.0
            bruh2 = np.expand_dims(bruh2, axis=0)
            predictions=model.predict(bruh2)
            print(CATAGORIES[int(np.argmax(predictions))])
            print(predictions(bruh2))
            print((np.argmax(predictions)))
            model.evaluate(X, y, verbose=2)
            value = input(hehe)
            value = int(value)
    except Exception as e:
       print("error")
       os.system('cls')
       model.evaluate(X, y, verbose=2)
       value = input(hehe)
       value = int(value)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D ,MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

model.fit(X, y, batch_size=256, epochs=15, validation_split=0.5)

predictions=model.predict([X])

value = input("Please enter an integer:\n")
value = int(value)

while value!=-1:
    
    try :
        print(CATAGORIES[int(np.argmax(predictions[value]))])
        plt.imshow(X[value], cmap=plt.cm.binary)
        plt.show()

        value = input("Please enter an integer:\n")
        value = int(value)
    except Exception as e:
        pass

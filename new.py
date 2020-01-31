import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test)=mnist.load_data() #28x28 pixel

#x_train = tf.keras.utils.normalize(x_train,axis=1) 
#x_test = tf.keras.utils.normalize(x_test,axis=1) 

model = Sequential()

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("relu"))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train , epochs=1)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

predictions=model.predict([x_test])

print(np.argmax(predictions[108]))
plt.imshow(x_test[108], cmap=plt.cm.binary)
plt.show()

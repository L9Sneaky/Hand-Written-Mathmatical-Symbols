import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


inputos = "E:/Books/Project - Operation Lettuce/PythonApplication1/New folder"
DATADIR = inputos;
CATAGORIES = ["alpha","beta","sigma","pi"]

print(os.listdir(inputos))

for category in CATAGORIES:
    path = os.path.join(DATADIR,category) #path to dir 
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))


IMG_SIZE = 45
#plt.imshow(new_array)
#plt.show()
#print(img_array)

training_data = []
def create_training_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR,category) #path to dir
        class_num = CATAGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))
random.shuffle(training_data)

#for sample in training_data[:10]:
#   print(sample[1])


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)#.reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = np.array(y)

print(len(X))
print(len(y))

pickle_out = open("X.pickle" , "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle" , "wb")
pickle.dump(y, pickle_out)
pickle_out.close() 
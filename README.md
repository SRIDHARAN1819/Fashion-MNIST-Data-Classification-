# Fashion-MNIST-Data-Classification-


Project - DEEP LEARNING PROJECT FASHION MNIST CLASSIFICATION

•ABSTRACT 

•OBJECTIVE

•INTRODUCTION 

•METHODOLOGY 

•CODE

•CONCLUSION 













ABSTRACT           

Computer vision is a widely researched area which makes it possible for computers to interact with images just like a human being will do. The advantage computer machines have is to be able to do this interaction on even large datasets at the speed of light, exposing features and ideas the normal human eyes would have taken years of hard work to achieve. For computers to complete visual tasks, machine and deep learning processes are usually programmed for the computer. Programming these tasks is a complex activity that requires patience and a good understanding of the subject matter. This work aims to develop a deep learning-based image multi-class classifier  to classify features and  create a more appealing visual and textual representation of the results using various tools and techniques. The model developed will be trained and tested on the Fashion MNIST dataset, and we will try to make a comparison of results obtained by also trying the model on the original MNIST dataset.
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

OBJECTIVE

The Fashion-MNIST clothing classification problem is a new standard dataset used in computer vision and deep learning. Although the dataset is relatively simple, it can be used as the basis for learning and practicing how to develop, evaluate, and use deep convolutional neural networks for image classification from scratch. This includes how to develop a robust test harness for estimating the performance of the model, how to explore improvements to the model, and how to save the model and later load it to make predictions on new data.


















INTRODUCTION

Computer image recognition is the ability of a  computer  to  recognize  an  image  and successfully classify it under a given label. It is  a  breakthrough  in  technology  that simplifies the way we think and interact with images  on  our  computers  and  over  the internet.  Also,  the  software  can  see differences between closely related  images, thereby making it easier and faster to make informed classifications and reports.  It  is  important  to  note  that  image classification  is  a  supervised  learning problem  in  that  target  classes  are  defined before  training  a  model  to  recognize  each defined target class and thereafter the model is  tested  with  sample  images  (Google Developers, n.d.). For  computer  vision models  to  be  able  to apply  more  flexibility  in  the  modelling process, It became necessary to apply more features  like  textures,  colors  and  shapes derived from the image data, making it more difficult to tune the models accurately since more input considerations need to be made. With  the  introduction  of  Convolutional Neural  Networks  (CNNs)  in  image classification, the CNNs just take the image data as is and learn how to extract the features to determine what class to place the image.   The objective of this work is to build a deep learning-based multi-class classifier on top of a given  image dataset  and provide insights into  the  dataset,  present  the  results  of  the model  developed  and  recommend  future improvements.














METHODOLOGY

Dataset

The FASHION_MNIST dataset was developed by Zalando as an alternative to the MNIST dataset which is usually utilized by Machine Learning, Artificial Intelligence and Data Science Engineers as a benchmark for validating their algorithms. This is also because the original MNIST dataset has been overused by machine learning researchers and is too easy for machine learning algorithms to achieve above 99% (Xiao, Rasul and Vollgraf, 2017).







Content

The dataset consists of images of fashion items with about 60,000 training sample images/labels and 10,000 test sample images/labels. Each sample image is a 28 by 28 grayscale image with 10 different labels.

![image](https://user-images.githubusercontent.com/82379566/200105891-0825c59a-410e-4685-9224-249d10ada0e3.png)

                     Figure 1. Image of sample fashion MNIST Dataset
                     
             Image Labels
             
Each test and training sample is assigned a label as described in the table below (Xiao, Rasul and Vollgraf, 2017):

Label	Description

0	T-shirt/top

1	Trouser

2	Pullover

3	Dress

4	Coat

5	Sandal

6	Shirt

7	Sneaker

8	Bag

9	Ankle boot

             Table 1. Feature classification for Fashion MNIST Dataset
             
             
 CODE

Step 1) Import Libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import keras

Step 2) Load data

(X_train, y_train), (X_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()

![image](https://user-images.githubusercontent.com/82379566/200106353-8d54497f-132d-4fe2-8f42-a6008f7f8f83.png)

# Print the shape of data

X_train.shape,y_train.shape, "***************" , X_test.shape,y_test.shape

![image](https://user-images.githubusercontent.com/82379566/200106381-046fa13f-bde5-4e64-b178-e064a1036035.png)

X_train[0]

![image](https://user-images.githubusercontent.com/82379566/200106399-d3d7b495-2412-4767-8e54-7dc36231013b.png)

y_train[0]

![image](https://user-images.githubusercontent.com/82379566/200106411-376d22b9-4b32-4adf-9410-9637d21a7594.png)

class_labels = [	"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",	"Sneaker",	"Bag",	"Ankle boot"]

class_labels

![image](https://user-images.githubusercontent.com/82379566/200106432-0887bf2f-38e0-4f8b-adf9-996a1c50e61d.png)

# show image

plt.imshow(X_train[0],cmap='Greys')

![image](https://user-images.githubusercontent.com/82379566/200106438-acf00268-e9c6-4910-a28f-87c29e87bad3.png)

plt.figure(figsize=(16,16))

j=1

for  i in np.random.randint(0,1000,25):

  plt.subplot(5,5,j);j+=1
  
  plt.imshow(X_train[i],cmap='Greys')
  
  plt.axis('off')
  
  plt.title('{} / {}'.format(class_labels[y_train[i]],y_train[i]))

![image](https://user-images.githubusercontent.com/82379566/200106474-5f9380e6-5048-4783-8b9e-08aae51b85bb.png)

X_train.ndim

![image](https://user-images.githubusercontent.com/82379566/200106489-936ef730-4c80-4447-a201-7eaab3d1d0e5.png)

X_train = np.expand_dims(X_train,-1)

X_train.ndim

![image](https://user-images.githubusercontent.com/82379566/200106501-f1c7ddcc-70d0-4f30-9d98-c26e4f275069.png)

X_test=np.expand_dims(X_test,-1)

# feature scaling

X_train = X_train/255

X_test= X_test/255

# Split dataset

from sklearn.model_selection import  train_test_split

X_train,X_Validation,y_train,y_Validation=train_test_split(X_train,y_train,test_size=0.2,random_state=2020)

X_train.shape,X_Validation.shape,y_train.shape,y_Validation.shape

![image](https://user-images.githubusercontent.com/82379566/200106528-ae111024-37bd-4d65-926b-3dff6b49074f.png)

Step 3) Buiding the CNN model

model=keras.models.Sequential([

                         keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),padding='valid',activation='relu',input_shape=[28,28,1]),
                         
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         
                         keras.layers.Flatten(),
                         
                         keras.layers.Dense(units=128,activation='relu'),
                         
                         keras.layers.Dense(units=10,activation='softmax')
                         
])

model.summary()

![image](https://user-images.githubusercontent.com/82379566/200106559-495de620-75b4-4143-bd60-7b0d0997a1d2.png)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,batch_size=512,verbose=1,validation_data=(X_Validation,y_Validation))

![image](https://user-images.githubusercontent.com/82379566/200106578-5022ac46-1320-47f4-a7d8-062b7742e681.png)

y_pred = model.predict(X_test)

y_pred.round(2)

![image](https://user-images.githubusercontent.com/82379566/200106596-696fc303-3187-443a-950a-9744963361ae.png)

y_test

![image](https://user-images.githubusercontent.com/82379566/200106612-e65552fb-26e5-489d-9cb8-829e618fce83.png)

model.evaluate(X_test, y_test)

![image](https://user-images.githubusercontent.com/82379566/200106622-74006270-231e-44b0-8314-554344ec4f05.png)

plt.figure(figsize=(16,16))
 
j=1

for i in np.random.randint(0, 1000,25):

  plt.subplot(5,5, j); j+=1
  
  plt.imshow(X_test[i].reshape(28,28), cmap = 'Greys')
  
  plt.title('Actual = {} / {} \nPredicted = {} / {}'.format(class_labels[y_test[i]], y_test[i], class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
  
  plt.axis('off')

![image](https://user-images.githubusercontent.com/82379566/200106644-f478e122-1335-4495-9ae6-1e19094388e2.png)

plt.figure(figsize=(16,30))
 
j=1

for i in np.random.randint(0, 1000,60):

  plt.subplot(10,6, j); j+=1
  
  plt.imshow(X_test[i].reshape(28,28), cmap = 'Greys')
  
  plt.title('Actual = {} / {} \nPredicted = {} / {}'.format(class_labels[y_test[i]], y_test[i], class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
  
  plt.axis('off')

![image](https://user-images.githubusercontent.com/82379566/200106668-6dbc9d22-2830-4d17-ac35-b355539b65e4.png)

"""## Confusion Matrix"""
 
![image](https://user-images.githubusercontent.com/82379566/200106675-f2680fb6-3ce5-4596-912e-e2d5467df752.png)

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(16,9))

y_pred_labels = [ np.argmax(label) for label in y_pred ]

cm = confusion_matrix(y_test, y_pred_labels)

sns.heatmap(cm, annot=True, fmt='d',xticklabels=class_labels, yticklabels=class_labels)
 
from sklearn.metrics import classification_report

cr= classification_report(y_test, y_pred_labels, target_names=class_labels)

print(cr)

![image](https://user-images.githubusercontent.com/82379566/200106723-133bc0f5-0b15-43fb-a175-7789ae12fea9.png)

"""# Save Model"""

![image](https://user-images.githubusercontent.com/82379566/200106728-9268b64c-24df-418f-9106-92e762f25a5e.png)

model.save('fashion_mnist_cnn_model.h5')

Build 2 complex CNN


#Building CNN model

cnn_model2 = keras.models.Sequential([

                         keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid',activation= 'relu', input_shape=[28,28,1]),
                         
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         
                         keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
                         
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         
                         keras.layers.Flatten(),
                         
                         keras.layers.Dense(units=128, activation='relu'),
                         
                         keras.layers.Dropout(0.25),
                         
                         keras.layers.Dense(units=256, activation='relu'),
                         
                         keras.layers.Dropout(0.25),
                         
                         keras.layers.Dense(units=128, activation='relu'),
                         
                         keras.layers.Dense(units=10, activation='softmax')
                         
                         ])
 
# complie the model

cnn_model2.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
 
#Train the Model

cnn_model2.fit(X_train, y_train, epochs=20, batch_size=512, verbose=1, validation_data=(X_Validation, y_Validation))
 
cnn_model2.save('fashion_mnist_cnn_model2.h5')
 
"""######## very complex model"""
 
#Building CNN model

cnn_model3 = keras.models.Sequential([

                         keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='valid',activation= 'relu', input_shape=[28,28,1]),
                         
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         
                         keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
                         
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         
                         keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
                         
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         
                         keras.layers.Flatten(),
                         
                         keras.layers.Dense(units=128, activation='relu'),
                         
                         keras.layers.Dropout(0.25),
                         
                         keras.layers.Dense(units=256, activation='relu'),
                         
                         keras.layers.Dropout(0.5),
                         
                         keras.layers.Dense(units=256, activation='relu'),
                         
                         keras.layers.Dropout(0.25),                        
                         
                         keras.layers.Dense(units=128, activation='relu'),
                         
                         keras.layers.Dropout(0.10),  
                         
                         keras.layers.Dense(units=10, activation='softmax')
                         
                         ])
 
# complie the model

cnn_model3.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy']) 

#Train the Model

cnn_model3.fit(X_train, y_train, epochs=50, batch_size=512, verbose=1, validation_data=(X_Validation, y_Validation))
 
cnn_model3.save('fashion_mnist_cnn_model3.h5')
 
cnn_model3.evaluate(X_test, y_test)

![image](https://user-images.githubusercontent.com/82379566/200106824-ccf16d95-e374-4398-960e-31b5cbad8f53.png)

CONCLUSION

In this work, we implemented a convolutional neural network model and used it to classify the fashion MNIST dataset. The model was then evaluated and the results were compared against the popular MNIST dataset for which the fashion variant is a drop-in replacement.The resulting accuracy obtained for the fashion MNIST was good but applying the same model to the original MNIST dataset also revealed how super easy it was to achieve an inch-perfect percentage accuracy. This in turn also validates the reason for the creation of the fashion variant of the dataset in that the original MNIST has become so easy and consequently overused.



URLs:

1.GitHub URL:

https://github.com/SRIDHARAN1819/Fashion-MNIST-Data-Classification-







import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape, LSTM
import matplotlib.pyplot as plt

#Import dataset and normalize to [0,1]
#Has shape (num_samples, 28, 28)
(data_train, labels_train), (data_test, labels_test) = fashion_mnist.load_data()
#fashion_mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0
data_train = data_train.reshape(60000, 28, 28, 1)
data_test = data_test.reshape(10000, 28, 28, 1)

data_train = data_train[0:2048, :, :, :]
data_test = data_test[0:512, :, :, :]
#Create labels as one-hot vectors
#labels_train and labels_test have shapes (60000, 10) and (10000 10,) respectively
labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)

labels_train = labels_train[0:2048, :]
labels_test = labels_test[0:512, :]
#Create and train model architecture
def CNN_overfit():
    #Easiest way to build model in Keras is using Squential. It allows models to 
    #be built layer by layer as we will do here
    model = Sequential()
      
    '''First hidden layer: a 2-dimensional convolutional layer with 256 feature maps and a 3×3 filter size'''
    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    '''Second hidden layer: a 2 × 2 max-pooling layer'''
    model.add(MaxPooling2D((2, 2)))
    '''Third hidden layer: a 2-dimensional convolutional layer with 128 feature maps and a 3×3 filter size'''
    model.add(Conv2D(128, (3, 3), activation='relu'))
    '''Fourth hidden layer: a 2 × 2 max-pooling layer'''
    model.add(MaxPooling2D((2, 2)))
    '''Fifth hidden layer: a layer to flatten the data'''
    model.add(Flatten())
    '''Sixth hidden layer: A dense (fully-connected) layer consisting of 100 perceptrons'''
    model.add(Dense(100, activation='relu'))
    '''Seventh hidden layer: A dense (fully-connected) layer consisting of 100 perceptrons'''
    model.add(Dense(100, activation='relu'))
    '''Output layer (classification probabilites): 10 perceptrons'''
    model.add(Dense(10, activation='softmax'))

    return model


#Create instance of CNN model graph
CNN_overfit = CNN_overfit()

#Compile model using an appropriate loss and optimizer algorithm
CNN_overfit.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model and assign training meta-data to a variable
cnn_model_mdata = CNN_overfit.fit(data_train, labels_train, 
                                  validation_data=(data_test, labels_test),
				  epochs=10, batch_size=128, shuffle=True)

#Print accuracy of model on testing set after training 
scores = CNN_overfit.evaluate(data_test,  labels_test)
print("Accuracy: %2f%%" %(scores[1]*100))


#Plot accuracy vs epoch
plt.plot(cnn_model_mdata.history['accuracy'])
plt.plot(cnn_model_mdata.history['val_accuracy'])
plt.title('CNN Accuracy vs. Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#Plot loss vs epoch
plt.plot(cnn_model_mdata.history['loss'])
plt.plot(cnn_model_mdata.history['val_loss'])
plt.title('CNN Loss vs. Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from sklearn.metrics import confusion_matrix
import seaborn as sns

#keras built ontop of tensorflow.

(train_data, train_label), (test_data, test_label) = tfk.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#visualization of data
print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

#printing function to make 'graphs' that generate the images.
def plot_fashion_mnist_dataset(data, label, prediction, prediction_flag=False):
    plt.figure(figsize=(15,15))
    for it in range(16):
        i = np.random.randint(len(data))
        plt.subplot(4,4,it+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i],cmap=plt.cm.binary)
        plt.xlabel(f"Label: {class_names[label[i]]}\n")
        if prediction_flag:
            plt.title(f"\n Prediction: {class_names[prediction[i]]}")
    plt.show()

#plot_fashion_mnist_dataset(train_data, train_label, None) #<- Uncomment me to see an example of the dataset visually.

#Pre-processing.
x_train = train_data/255 #converts to 0 or 1.
x_test = test_data/255 

#Expand the dimensions to work for cnn.
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

#splitting training and validation sets.
x_val, y_val = x_train[50000:,:], train_label[50000:] #validation
x_train, y_train = x_train[:50000,:],  train_label[:50000] #training

print(x_train.shape)

#Defining the model.
model = tfk.Sequential([
    #input layer
    tfk.Input(shape=x_train[0].shape, name="input"),

    #First convolutional layer followed by max_pooling.
    tfk.layers.Conv2D(32,(3,3),strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    #Second convolutional layer, followed by max pooling.
    tfk.layers.Conv2D(64,(3,3),strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    #Third convolutional layer, followed by max pooling.
    tfk.layers.Conv2D(128,(3,3),strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    #Neural Network portion...
    tfk.layers.Flatten(),
    tfk.layers.Dense(256,activation="relu",name="dense_1"), #issue on this line with cardinality.
    tfk.layers.Dense(10,activation="softmax",name="dense_2"),  
])

#training the model.
#step 1. optimizer selection.
optimizer = tfk.optimizers.Adam(learning_rate=0.001)

#step 2. Choose metrics.
metric = tfk.metrics.SparseCategoricalAccuracy() #standard on classification models gives a value of 1 for correct, 0 for incorrect.

#step 3. loss function.
loss_function = tfk.losses.SparseCategoricalCrossentropy() #

#step 4. compile the network.
model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)

#trains the model at this point, keeping track of stats utilizing the validation set
history = model.fit(x_train, y_train, batch_size = 64, epochs=10, validation_data=(x_val,y_val),)


#finally, plot how things went.
plt.figure(figsize=(7,4),dpi=100)
plt.plot(history.history['loss']) #history contains dictionaries for different types of data.
plt.plot(history.history['val_loss'])
plt.xlabel('epoch',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.legend(['training loss','validation loss'])
plt.show()

print("Evaluate on test data")
results = model.evaluate(x_test, test_label, batch_size=128)
print("test loss, test acc:",results)


pred= model.predict(x_test).argmax(axis=1)
plot_fashion_mnist_dataset(test_data, test_label, pred, prediction_flag=True)
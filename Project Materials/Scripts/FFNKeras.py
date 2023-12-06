import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

#keras built ontop of tensorflow.
def loadData():
    '''Important! You need to run the image preparer at least once before running this file!'''
    print("Loading Files. Some files are incredibly large and will take some time to load.")
    with open('training_data.pickle','rb') as f:
        training_data = pickle.load(f)
    print("Loaded training_data.pickle")

    with open('training_labels.pickle','rb') as f:
        training_labels = pickle.load(f)
    print("Loaded training_labels.pickle")

    with open('testing_data.pickle','rb') as f:
        testing_data = pickle.load(f)
    print("Loaded testing_data.pickle")

    with open('testing_labels.pickle','rb') as f:
        testing_labels = pickle.load(f)
    print("Loaded testing_labels.pickle")
    print("Loading complete. Beginning conversion...")
    training_data = list(training_data.values())[0:2000]  #Limits data input. Much much faster without taking the whole set.
    training_labels = list(training_labels.values())[0:2000]
    testing_data = list(testing_data.values())[0:500]
    testing_labels = list(testing_labels.values())[0:500]
    print("Conversion completed.")
    return (training_data,training_labels),(testing_data,testing_labels)

(train_data, train_label), (test_data, test_label) = loadData()



train_label = np.array(train_label)
test_label = np.array(test_label)
train_data = np.array(train_data)
test_data = np.array(test_data)



model = models.Sequential()
model.add(layers.Dense(4900, activation='relu', input_shape=(1,4900)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2))
model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(name='mean_squared_error'),
            metrics=['accuracy'])
model.summary()

history = model.fit(train_data, 
                    train_label, 
                    epochs=2, 
                    validation_data=(test_data, 
                                     test_label))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
print(test_acc)

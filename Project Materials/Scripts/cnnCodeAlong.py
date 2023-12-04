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
    training_data = list(training_data.values())
    training_labels = list(training_labels.values())
    testing_data = list(testing_data.values())
    testing_labels = list(testing_labels.values())
    print("Conversion completed.")
    return (training_data,training_labels),(testing_data,testing_labels)

def addRGBAndReshape(data):
    #inputs are pre-flattened. A row is 4900 pixels. there is 100,000 pictures to convert. (100000,4900)
    rgb_data = np.empty(shape=(len(data),70,70,3))
    count = 0
    for row in data:
        new_picture = np.empty(shape=(70,70,3)) #A single row is one picture.
        for i in range(70):
            picture_row = row[i*70:(i+1)*70]
            current_row=np.empty(shape=(70,3))
            for pixel in picture_row:
                if pixel == 1:
                    current_row = np.array[1,1,1]
                    # current_row=np.concatenate(current_row,np.array([1,1,1]))
                else:
                    current_row = np.array[0,0,0]
                    # current_row=np.concatenate(current_row,np.array([0,0,0]))
            # new_picture.concatenate(np.array(current_row))
            new_picture[i] = current_row
        rgb_data = np.concatenate(np.array(new_picture))
        count+=1
        print(count,"Images Processed")
    print("All Images Processed")
    input(rgb_data.shape)
    return np.array(rgb_data)

def addRGBAndReshapeV2(data):
    reshaped_data = np.zeros(shape=(len(data),70,70,3),dtype=int)
    for row_count in range(len(data)):
        #A row is 4900 pixels.
        current_row = data[row_count]
        for x_pointer in range(70):
            #offset within section.
            for y_pointer in range(70):
                #offset to get to section.
                if(current_row[y_pointer*70+x_pointer]==1):
                    reshaped_data[row_count][y_pointer][x_pointer][0]=1
                    reshaped_data[row_count][y_pointer][x_pointer][1]=1
                    reshaped_data[row_count][y_pointer][x_pointer][2]=1
                else:
                    reshaped_data[row_count][y_pointer][x_pointer][0]=0
                    reshaped_data[row_count][y_pointer][x_pointer][1]=0
                    reshaped_data[row_count][y_pointer][x_pointer][2]=0
                #Very gross, but, it will be a np.ndarray for sure!
        print(row_count,"Images Processed")
    print("Done.")
    return reshaped_data



def showDataSample(data,labels,classes):
    #Show data loaded in! Borrowing example data display from tensorflow example. 
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i]*255)#multiply by 255 to un-normalize the images.
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(classes[labels[i][0]])
    plt.show()

(train_data, train_label), (test_data, test_label) = loadData()
class_names = ['Not Visible','Visible']
#Current shape we have is 70,70 per image. Ideally we'll want to have 70,70,3 for the shape. with the 3 representing the rgb colors for a given pixel.
print("Re-adding color depth.")
train_data = addRGBAndReshapeV2(train_data)
test_data = addRGBAndReshapeV2(test_data)
train_label = np.array(train_label)
test_label = np.array(test_label)
print("Done")

showDataSample(train_data,train_label,class_names)

singlePicture = train_data[0]
input(singlePicture.shape)
input(singlePicture)

model = models.Sequential()
model.add(layers.Conv2D(70, (3, 3), activation='relu', input_shape=(70, 70, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2))
model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(name='mean_squared_error'),
            metrics=['accuracy'])
model.summary()

history = model.fit(train_data, 
                    train_label, 
                    epochs=10, 
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
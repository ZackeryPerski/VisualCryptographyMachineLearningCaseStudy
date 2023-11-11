import random
#from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import urllib.request
import gzip
import struct
import pickle
import threading


np.random.seed(42)
images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

    
def load_mnist(images_url, labels_url):
    with urllib.request.urlopen(images_url) as f_images, gzip.GzipFile(fileobj=f_images) as f_images_gzip:
        magic, num, rows, cols = struct.unpack(">IIII",f_images_gzip.read(16))
        images = np.frombuffer(f_images_gzip.read(), dtype=np.uint8).reshape(num,784)

    with urllib.request.urlopen(labels_url) as f_labels, gzip.GzipFile(fileobj=f_labels) as f_labels_gzip:
        magic,num = struct.unpack(">II", f_labels_gzip.read(8))
        labels = np.frombuffer(f_labels_gzip.read(),dtype=np.uint8)

    return images,labels

def pattern_generator():
    pixels_top =    [(0,0),(1,1),(1,0),(0,1),(0,1),(1,0)]
    pixels_bottom = [(1,1),(0,0),(1,0),(0,1),(1,0),(0,1)]
    choice = np.random.randint(6)
    return pixels_top[choice], pixels_bottom[choice]

def create_overlays(s2_prime, s1, x_offset, y_offset):
    #Extra gouda cheese here
    invalid_x_offset = list(range(7))
    invalid_x_offset.remove(x_offset)
    invalid_x_offset = random.choice(invalid_x_offset)

    invalid_y_offset = list(range(7))
    invalid_y_offset.remove(y_offset)
    invalid_y_offset = random.choice(invalid_y_offset)
    #initialize the working copies.
    valid_overlay = []
    invalid_overlay = []
    #force deep copy construction argh!
    for i in range(len(s2_prime)):
        row = []
        for j in range(len(s2_prime[0])):
            row.append(s2_prime[i][j])
        valid_overlay.append(row)
    for i in range(len(s2_prime)):
        row = []
        for j in range(len(s2_prime[0])):
            row.append(s2_prime[i][j])
        invalid_overlay.append(row)

    #overlays are made by pixel wise xoring. Black pulls down.
    for i in range(len(s1)):
        for j in range(len(s1[0])):
            pos_x_valid = i+2*x_offset
            pos_y_valid = j+2*y_offset
            pos_x_invalid = i+2*invalid_x_offset
            pos_y_invalid = j+2*invalid_y_offset
            if(s1[i][j]==0):
                valid_overlay[pos_x_valid][pos_y_valid] = 0
                invalid_overlay[pos_x_invalid][pos_y_invalid] = 0
    #overlays generated.
    return valid_overlay, invalid_overlay


def produce_encrypted_shares(image):
    s1, s2 = [],[]
    for row in image:
        #when encoding the images, we'll use a simple 1 -> 2x2 cipher.
        s1_row1 = []
        s1_row2 = []
        s2_row1 = []
        s2_row2 = []
        #in 2x2, there are always 2 pixels filled. A black pixel is represented by the 2 shares having complementary sets of 0.
        # 0 0  |  1 1 |  0 1  |  1 0  |  1 0  |  0 1
        # 1 1  |  0 0 |  1 0  |  0 1  |  1 0  |  0 1
        for pixel in row:
            pattern_top, pattern_bottom = pattern_generator()
            pixel_tl, pixel_tr = pattern_top
            pixel_bl, pixel_br = pattern_bottom
            s1_row1.append(pixel_tl)
            s1_row1.append(pixel_tr)
            s1_row2.append(pixel_bl)
            s1_row2.append(pixel_br)

            if pixel == 1: #white #On white, the pixel pattern on s2 is the same as s1.
                s2_row1.append(pixel_tl)
                s2_row1.append(pixel_tr)
                s2_row2.append(pixel_bl)
                s2_row2.append(pixel_br)
            else: #on black, s2 has the complement.
                s2_row1.append(abs(pixel_tl-1))
                s2_row1.append(abs(pixel_tr-1))
                s2_row2.append(abs(pixel_bl-1))
                s2_row2.append(abs(pixel_br-1))
        s1.append(s1_row1)
        s1.append(s1_row2)
        s2.append(s2_row1)
        s2.append(s2_row2)
    return s1, s2

def produce_s2_prime(image_share):
    s2_prime = []
    for i in range(35):
        row1 = []
        row2 = []
        for j in range(35):
            pattern_top, pattern_bottom = pattern_generator()
            pixel_tl, pixel_tr = pattern_top
            pixel_bl, pixel_br = pattern_bottom
            row1.append(pixel_tl)
            row1.append(pixel_tr)
            row2.append(pixel_bl)
            row2.append(pixel_br)
        s2_prime.append(row1)
        s2_prime.append(row2)
    #s2_prime background is generated. time to overlay s2 properly.
    #s2_prime is 70x70. s2 is 56x56. = 14x14 difference. valid movements however means that this is actually 7x7.
    #Therefore we generate two offset numbers, from 0-7(exclusive.)
    x_offset = np.random.randint(7)
    y_offset = np.random.randint(7)

    #imprint s2 onto s2_prime to actually finish s2_prime. 
    for i in range(len(image_share)):
        for j in range(len(image_share[0])):
            s2_prime[i+2*x_offset][j+2*y_offset] = image_share[i][j]
    
    return s2_prime, x_offset, y_offset
#pixels are represented currently as RGB format with 0,0,0 as black. or 1,1,1 as white. First part of preparation will be to convert to smaller more meaningful data.
#https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder

#preparations
images, labels = load_mnist(images_url,labels_url)
train_images = images[:50000]/255.0 #normalize the images
test_images = images[50000:]/255.0 #normalize the images

imageCount = 0
training_dict_data = {}
training_dict_labels = {}
test_dict_data = {}
test_dict_labels = {}


for image in train_images:
    converted_image = np.array(image)#convert to np.array object.
    converted_image = np.ceil(converted_image)
    converted_image = converted_image.reshape(-1,28)#reshape mnist image back into a matrix for conversion.
    s1, s2 = produce_encrypted_shares(converted_image)
    #S1,S2 generated. Time to create the S2' that hides S2. Not optimized, but ease over speed at the moment.
    s2_prime, x_offset, y_offset = produce_s2_prime(s2)
    
    #create two samples for training. One valid overlay, one invalid.
    valid_overlay, invalid_overlay = create_overlays(s2_prime, s1, x_offset, y_offset)

    #Used for showcasing what's being generated.
    '''
    plt.matshow(invalid_overlay)
    plt.matshow(valid_overlay)
    plt.show()
    '''

    #now that all manipulation is over, re-flatten the matrix.
    valid_overlay = np.array(valid_overlay).flatten()
    invalid_overlay = np.array(invalid_overlay).flatten()
    #print("Shape of valid_overlay:",valid_overlay.shape)
    #print("Shape of invalid_overlay:",invalid_overlay.shape)
    #input()

    #export json. Choose randomly if correct overlay or incorrect is saved first to prevent true/false patterning in the data.
    if(np.random.randint(2)==0):
        training_dict_data["image_"+str(imageCount)] = valid_overlay.tolist()
        training_dict_labels["label_"+str(imageCount)] = [1,0] #hot encode true.
        imageCount+=1
        training_dict_data["image_"+str(imageCount)] = invalid_overlay.tolist()
        training_dict_labels["label_"+str(imageCount)] = [0,1] #hot encode false.
        imageCount+=1
    else:
        training_dict_data["image_"+str(imageCount)] = invalid_overlay.tolist()
        training_dict_labels["label_"+str(imageCount)] = [0,1] #hot encode false.
        imageCount+=1
        training_dict_data["image_"+str(imageCount)] = valid_overlay.tolist()
        training_dict_labels["label_"+str(imageCount)] = [1,0] #hot encode true.
        imageCount+=1
    print(imageCount,"Images produced.")
    
#at this point training images and labels have been produced.
#time to create testing images and labels.
imageCount = 0

for image in test_images:
    converted_image = np.array(image)#convert to np.array object.
    converted_image = np.ceil(converted_image)
    converted_image = converted_image.reshape(-1,28)#reshape mnist image back into a matrix for conversion.
    s1, s2 = produce_encrypted_shares(converted_image)
    #S1,S2 generated. Time to create the S2' that hides S2. Not optimized, but ease over speed at the moment.
    s2_prime, x_offset, y_offset = produce_s2_prime(s2)
    
    #create two samples for testing. One valid overlay, one invalid.
    valid_overlay, invalid_overlay = create_overlays(s2_prime, s1, x_offset, y_offset)

    #now that all manipulation is over, re-flatten the matrix.
    valid_overlay = np.array(valid_overlay).flatten()
    invalid_overlay = np.array(invalid_overlay).flatten()
    #decide which one to store in testing. only using one!
    if(np.random.randint(2)==0):
        test_dict_data["image_"+str(imageCount)] = valid_overlay.tolist()
        test_dict_labels["label_"+str(imageCount)] = [1,0] #hot encode true.
        imageCount+=1
    else:
        test_dict_data["image_"+str(imageCount)] = invalid_overlay.tolist()
        test_dict_labels["label_"+str(imageCount)] = [0,1] #hot encode false.
        imageCount+=1
    print(imageCount,"Images produced.")

print("Dumping files. This will take some time.")

with open('training_data.pickle','wb') as handle:
    pickle.dump(training_dict_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
print('Training data successfully dumped.')

with open('training_labels.pickle', 'wb') as handle:
    pickle.dump(training_dict_labels,handle,protocol=pickle.HIGHEST_PROTOCOL)
print('Training labels successfully dumped.')

with open('testing_data.pickle','wb') as handle:
    pickle.dump(test_dict_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
print('Testing data successfully dumped.')

with open('testing_labels.pickle', 'wb') as handle:
    pickle.dump(test_dict_labels,handle,protocol=pickle.HIGHEST_PROTOCOL)
print('Testing labels successfully dumped.')

print("All files dumped.")
                
                
                



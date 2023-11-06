import os, glob
#from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import urllib.request
import gzip
import struct

np.random.seed(42)
images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

#pathSouorce = '../Images/BW_Samples'
pathPrimeShares = '../Images/Prime Shares'
pathShares = '../Images/Shares'

'''
def one_hot(labels,num_classes):
    return np.eye(num_classes)[labels]
'''
    
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

#pixels are represented currently as RGB format with 0,0,0 as black. or 1,1,1 as white. First part of preparation will be to convert to smaller more meaningful data.
#https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder

#preparations
images, labels = load_mnist(images_url,labels_url)
train_images = images[:50000]/255.0 #normalize the images
test_images = images[50000:]/255.0 #normalize the images

for image in train_images:
    converted_image = np.array(image)#convert to np.array object.
    converted_image = np.ceil(converted_image)
    converted_image = converted_image.reshape(-1,28)#reshape mnist image back into a matrix for conversion.
    input(converted_image.shape)
    input(str(converted_image))
    s1, s2 = [],[]
    for row in converted_image:
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
    
    print(s1)
    s1 = np.array(s1)
    print(s1.shape)
    input("Waiting. S1 printed.")
    print(s2)
    s2 = np.array(s2)
    print(s2.shape)
    input("Waiting. S2 printed.")
    



                
                
                



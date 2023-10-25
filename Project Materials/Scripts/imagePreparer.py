import os, glob
#from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(42)
pathSouorce = '../Images/BW_Samples'
pathPrimeShares = '../Images/Prime Shares'
pathShares = '../Images/Shares'

def pattern_generator():
    pixels_top =    [(0,0),(1,1),(1,0),(0,1),(0,1),(1,0)]
    pixels_bottom = [(1,1),(0,0),(1,0),(0,1),(1,0),(0,1)]
    choice = np.random.randint(6)
    return pixels_top[choice], pixels_bottom[choice]

#pixels are represented currently as RGB format with 0,0,0 as black. or 1,1,1 as white. First part of preparation will be to convert to smaller more meaningful data.
#https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder
for filename in glob.glob(os.path.join(pathSouorce,'*.png')):
    current_image = image.imread(filename)
    print(current_image)
    print(current_image.dtype)
    print(current_image.shape)
    plt.imshow(current_image)
    plt.show()
    convert_to_simple = []
    for row in current_image:
        simple_row = []
        for pixel in row:
            if pixel[0] == 0:
                simple_row.append(0) #black
            else:
                simple_row.append(1) #white
        convert_to_simple.append(simple_row)
    print(convert_to_simple)
    current_image_simple = np.array(convert_to_simple)
    print(current_image_simple.shape)
    #at this point the np array is set to go for the algorithm.
    
    s1, s2 = [],[]
    for row in current_image_simple:
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
    



                
                
                



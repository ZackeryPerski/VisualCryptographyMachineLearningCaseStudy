import os, glob
#from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(42)
pathSouorce = '../Images/BW_Samples'
pathPrimeShares = '../Images/Prime Shares'
pathShares = '../Images/Shares'

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
    #print(convert_to_simple)
    current_image_simple = np.array(convert_to_simple)
    print(current_image_simple.shape)


import os, glob

pathSouorce = '../Images/BW_Samples'
pathPrimeShares = '../Images/Prime Shares'
pathShares = '../Images/Shares'

#https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder
for filename in glob.glob(os.path.join(pathSouorce,'*.png')):
    with open(os.path.join(os.getcwd(),filename),'r') as f:
        print(filename)
        
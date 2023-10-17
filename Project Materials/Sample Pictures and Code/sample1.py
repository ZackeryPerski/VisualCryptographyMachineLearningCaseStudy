#import math
#import math as m
#from math import ...


mylist = []
mydict = {}
for i in range(1,6):
    mylist.append(i*i)
    mydict[i] = i*i
print(mylist)
print(mydict)

#function exercise
def average(mylist):
    length = len(mylist)
    val = 0
    if length == 0:
        return 0 #probably a better returnable.
    for i in range(0,length):
        val+=mylist[i]
    return val/length

samplelist = [1,2,3,4,5]
print(average(samplelist))

permutations = []
permutationCount = 0
language = ['a','b','c']
for letter1 in language:
    for letter2 in language:
        for letter3 in language:
            for letter4 in language:
                permutations.append(letter1+letter2+letter3+letter4)
                permutationCount+=1

print(permutations)
print(permutationCount)

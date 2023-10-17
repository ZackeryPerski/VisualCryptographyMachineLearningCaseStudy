
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
'''Notes
x1, x2, x3 -> inputs
w1, w2, w3 -> weights
b -> bias

activation function

Classify if an email is spam or not
What inputs could we have to the neuron?
Keyword #1 frequency
Keyword #2 frequency

How do we find the correct weights and balances
Training is like teaching the computer to recognize patterns
the goal of training is to find the right 'recipe' of weights and biases to most accurately predict the outcome.

-5

sigmoid(x) = 1/1+e^-x
outputs values in range(0,1)
think of it as smooth switch
first email had initial output value of 8

sigmoid(8) = 0.9997 (probably spam)
second email
sigmoid(-1) = 0.2689(probably not spam)

rectified linear unit: ReLU
f(x) = max(0,x)
computentially efficient

'''
import math

def sigmoid(x):
    return 1/(1+math.e**-x)

def relu(x):
    return max(0,x)

def neuron(inputs,weights,bias):
    #assuming weights equal inputs.
    sigma = 0
    for i in range(0,len(inputs)):
        sigma+=inputs[i]*weights[i]
    return sigmoid(sigma+bias)



print(sigmoid(0))
print(sigmoid(4))
print(sigmoid(-4))

print(relu(5))
print(relu(-5))
print(relu(0))

inputs = [5,1,9]
weights = [0.2, .4, -.1]
bias = .1
result = neuron(inputs, weights, bias)
print(result)
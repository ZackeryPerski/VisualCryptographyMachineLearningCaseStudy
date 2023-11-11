import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

print("Loading JSON Files. Some files are incredibly large and will take some time to load.")
with open('training_data.json','r') as f:
    training_data = json.load(f)
print("Loaded training_data.json")

with open('training_labels.json','r') as f:
    training_labels = json.load(f)
print("Loaded training_labels.json")

with open('testing_data.json','r') as f:
    testing_data = json.load(f)
print("Loaded testing_data.json")

with open('testing_labels.json','r') as f:
    testing_labels = json.load(f)
print("Loaded testing_labels.json")
print("Loading complete.")

training_data = np.array(list(training_data.values()))
training_labels = np.array(list(training_labels.values()))
testing_data = np.array(list(testing_data.values()))
testing_labels = np.array(list(testing_labels.values()))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# One hot encoding for labels
def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]


def evaluate_metrics(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / total
    
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1
    

# Hyperparameters
input_size = 4900  # Each image: 28x28 pixels, total 784 elements
hidden_size = 128  # Hidden layer size
output_size = 2  # Number of classes (digits 0-9)
learning_rate = 0.01  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size

# Initialize weights and biases with random values
np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

training_history = []
evaluation_history = []
# Training
for epoch in range(epochs):
    y_true_train = []
    y_pred_train = []
    for i in range(0, len(training_data), batch_size):
        # Forward pass
        batch_x = training_data[i:i + batch_size]
        batch_y = training_labels[i:i + batch_size]
        h1 = np.dot(batch_x, w1) + b1
        a1 = sigmoid(h1)

        h2 = np.dot(a1, w2) + b2
        output = sigmoid(h2)

        # backward pass
        # loss function
        error = batch_y - output

        # Store true and predicted labels
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(batch_y, axis=1)
        y_true_train.extend(true_labels)
        y_pred_train.extend(predictions)

        # Compute gradients
        # Update the weights and biases

        g_out = error * sigmoid_derivative(output)
        w2 += learning_rate * a1.T.dot(g_out)
        b2 += np.sum(g_out, axis=0, keepdims=True) * learning_rate

        g_hidden = g_out.dot(w2.T) * sigmoid_derivative(a1)
        w1 += learning_rate * batch_x.T.dot(g_hidden)
        b1 += np.sum(g_hidden, axis=0, keepdims=True) * learning_rate

    accuracy_train, precision_train, recall_train, f1_train = evaluate_metrics(y_true_train, y_pred_train)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Training Metrics:")
    print(f"  Accuracy: {accuracy_train:.4f}")
    print(f"  Precision: {precision_train:.4f}")
    print(f"  Recall: {recall_train:.4f}")
    print(f"  F1-Score: {f1_train:.4f}")
    training_history.append(accuracy_train)
    #TODO: Run through the currrent FFN utilizing the Testing Set, and recording the overall accuracy.
    y_true_eval = []
    y_pred_eval = []
    for i in range(0, len(testing_data), batch_size):
        # Forward pass
        batch_x = testing_data[i:i + batch_size]
        batch_y = testing_labels[i:i + batch_size]

        h1 = np.dot(batch_x, w1) + b1
        a1 = sigmoid(h1)

        h2 = np.dot(a1, w2) + b2
        output = sigmoid(h2)

        # backward pass
        # loss function
        error = batch_y - output

        # Store true and predicted labels
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(batch_y, axis=1)
        y_true_eval.extend(true_labels)
        y_pred_eval.extend(predictions)
    accuracy_test, precision_test, recall_test, f1_test = evaluate_metrics(y_true_eval, y_pred_eval)
    evaluation_history.append(accuracy_test)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Evaluation Metrics:")
    print(f"  Accuracy: {accuracy_test:.4f}")
    print(f"  Precision: {precision_test:.4f}")
    print(f"  Recall: {recall_test:.4f}")
    print(f"  F1-Score: {f1_test:.4f}")


#Graphing Example
#finally, plot how things went.
plt.figure(figsize=(7,4),dpi=100)
plt.plot(training_history) #history contains dictionaries for different types of data.
plt.plot(evaluation_history)
plt.xlabel('epoch',fontsize=20)
plt.ylabel('accuracy',fontsize=20)
plt.legend(['training accuracy','validation accuracy'])
plt.show()
from neural import neuralNetwork
from data_read import *
import numpy as np
import matplotlib.pyplot

input_nodes = 784
hidden_nodes = [500]
output_nodes = 10
learning_rate = 0.2

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

data_list = images_read("train/train-images.gz")
np.random.shuffle(data_list)

for elem in data_list:
    all_values = elem.split(',')
    scaled_input = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(scaled_input, targets)

test_list = labels_read("train/train-labels.gz")

np.random.shuffle(test_list)
for elem in test_list:

    all_values = elem.split(',')
    scaled_input = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    correct_output = all_values[0]
    result = np.transpose(n.query(scaled_input))
    output_num = np.argmax(result)
    matplotlib.pyplot.imshow(scaled_input.reshape((28, 28)), cmap='Greys')
    matplotlib.pyplot.show()
    print(f"Полученно: {output_num} - должно: {all_values[0]}")
    input()
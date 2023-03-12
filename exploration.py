import itertools
from start import *
import matplotlib.pyplot as plt


def create_neurons():
    input_node = 784
    hidden_nodes = None
    output_nodes = 10
    learning_rate = 0.2

    images = images_read("train/train-images.gz")
    labels = labels_read("train/train-labels.gz")

    for level, quantity in itertools.product(range(1, 4), range(100, 1000, 100)):
        hidden_nodes = [quantity] * level
        print(hidden_nodes)
        neural = neuralNetwork(input_node, hidden_nodes, output_nodes, learning_rate)
        training(neural, images, labels)
        print("Ok")
        save_file(neural, file=f"exploration/{level}.{quantity}")


def exploration():
    images = images_read("train/train-images.gz")
    labels = labels_read("train/train-labels.gz")

    with open("exp", "w") as file:
        for level, quantity in itertools.product(range(1, 4), range(100, 1000, 100)):
            neural = use_saved_file(file=f"exploration/{level}.{quantity}")
            file.write(f"{level}.{quantity}:{correct(neural, images, labels, print_result=False)}\n")


def exploration_view():
    with open("exp", "r") as file:
        data = file.readlines()



if __name__ == "__main__":
    create_neurons()
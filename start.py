from data_read import *
from neural import neuralNetwork
import matplotlib.pyplot as plt


def main():
    input_node = 784
    hidden_nodes = [500, 500]
    output_nodes = 10
    learning_rate = 0.2

    neural = neuralNetwork(input_node, hidden_nodes, output_nodes, learning_rate)

    images = images_read("train/train-images.gz")
    labels = labels_read("train/train-labels.gz")

    training(neural, images, labels)
    view(neural, images, labels)


def training(neural, images, labels, pack_len=1):
    """обучение нейросети"""
    ln = [*range(len(images))]
    np.random.shuffle(ln)

    for elem in ln:
        image = image_to_data(images[elem])
        label = label_to_data(labels[elem], 10)
        neural.train(image, label)


def view(neural, images, labels):
    """презентация"""
    ln = [*range(len(labels))]
    np.random.shuffle(ln)

    for elem in ln:
        image = images[elem]
        label = labels[elem]

        result = np.transpose(neural.query(image_to_data(images[elem])))

        print(f"Результат: {np.argmax(result)}\n"
              f"Должно: {label}")

        fig, ax = plt.subplots()
        ax.pcolormesh(image, cmap=plt.colormaps["Greys"])
        plt.show()

def image_to_data(image: list[list[int]]) -> list[int]:
    return (np.reshape(image, -1) / 255 * 0.99) + 0.01
def label_to_data(label: int, ln: int) -> list[int]:
    data = np.zeros(ln) + 0.01
    data[label] = 0.99
    return data

if __name__ == '__main__':
    main()
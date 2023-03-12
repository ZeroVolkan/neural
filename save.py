from neural import *


def save_file(neural: neuralNetwork, file=None):
    name = file or input("Имя файла: ")
    with open(name, "w") as f:
        for elem in (neural.inodes, neural.onodes, neural.lr):
            f.write(f"{elem}\\")
        for elem in neural.hnodes:
            f.write(f"{elem},")

    np.savez(name, *neural.w)


def use_saved_file(file=None) -> neuralNetwork:
    name = file or input("Имя файла: ")
    with open(name, "r") as f:
        objs = f.read().split("\\")

    neural = neuralNetwork(int(objs[0]), list(map(int, objs[3].split(",")[:-1])), int(objs[1]), float(objs[2]))
    item = np.load(f"{name}.npz")
    neural.w = [item[i] for i in item]

    return neural
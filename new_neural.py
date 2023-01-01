import numpy as np
import scipy.special as sc # sc.expit == сигмоида

class Neural:
    def __init__(self, nodes: list[int], learning_rate: float=0.2):
        self.neural = [np.random.randint(-10, 10, size=(nodes[i + 1], nodes[i])) for i in range(len(nodes) - 1)]
        self.lr = learning_rate

    def look(self, inputs: list[int]) -> np.array:
        if len(inputs) != len(self.neural[0][0]):
            raise TypeError(f"Неправильный ввод!")

        for grop in range(len(self.neural)):
            inputs = np.array([sc.expit(np.dot(inputs, self.neural[grop][i])) for i in range(len(self.neural[grop]))], dtype=np.float32)
        return inputs



def test():
    Test = Neural([3, 24, 24])
    print(np.argmax(Test.look([12,12,12])))

if __name__ == "__main__":
    test()
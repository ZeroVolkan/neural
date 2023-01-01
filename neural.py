import numpy as np
import scipy.special as sc


class neuralNetwork:
    """нейронная сеть"""
    def __init__(self, input, hidden, output, learning_rate):
        # задаем количество узлов во входном, скрытом и выходном слое
        self.inodes = input
        self.hnodes = hidden
        self.onodes = output

        self.lr = learning_rate # коэффициент обучения
        self.w = self.create() # список матриц весовых коэфициентов

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: sc.expit(x)

    # создание списка матриц весовых коэфициентов
    def create(self):
        list_w = []
        if len(self.hnodes) == 0:
            list_w.append(np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes)))
        else:
            for i in range(len(self.hnodes)):
                if i == 0:
                    list_w.append(np.random.normal(0.0, pow(self.hnodes[i], -0.5), (self.hnodes[i], self.inodes)))
                    if len(self.hnodes) > 1:
                        list_w.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5), (self.hnodes[i + 1],
                                                                                        self.hnodes[i])))
                    else:
                        list_w.append(np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes[i])))
                elif i == len(self.hnodes) - 1:
                    list_w.append(np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes[i])))
                else:
                    list_w.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5), (self.hnodes[i + 1],
                                                                                        self.hnodes[i])))
        return list_w


    def train(self, inputs_list, targets_list):
        """метод для тренировки нейронной сети"""
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        outputs_list = []
        outputs = 0
        i = 0
        # получение выходных сигналов на каждом слое
        for hn in self.w:
            if i == 0:
                outputs = self.activation_function(np.dot(hn, inputs))
            else:
                outputs = self.activation_function(np.dot(hn, outputs))
            outputs_list.append(outputs)
            i += 1
        # получение ошибки в зависимости от целевого значения на выходе
        output_errors = targets - outputs
        # обратное распространение ошибки в зависимости от колличества скрытых слоев
        if len(self.hnodes) == 0:
            self.w[0] += self.lr * np.dot((output_errors * outputs_list[0] * (1.0 - outputs_list[0])),
                                     np.transpose(inputs))
        else:
            i = 0
            outputs_list = outputs_list[::-1]
            for out in outputs_list:
                if i == 0:
                    self.w[-(i + 1)] += self.lr * np.dot((output_errors * out * (1.0 - out)),
                                                  np.transpose(outputs_list[i + 1]))
                elif i == len(outputs_list) - 1:
                    if len(self.w) > 2:
                        errors = np.dot(self.w[1].T, errors)
                        self.w[0] += self.lr * np.dot((errors * out * (1.0 - out)), np.transpose(inputs))
                    elif len(self.w) == 2:
                        errors = np.dot(self.w[-1].T, output_errors)
                        self.w[-(i + 1)] += self.lr * np.dot((errors * out * (1.0 - out)), np.transpose(inputs))
                elif i == 1:
                    errors = np.dot(self.w[-1].T, output_errors)
                    self.w[-(i + 1)] += self.lr * np.dot((errors * out * (1.0 - out)),
                                                         np.transpose(outputs_list[i + 1]))
                else:
                    errors = np.dot(self.w[-i].T, errors)
                    self.w[-(i + 1)] += self.lr * np.dot((errors * out * (1.0 - out)),
                                                         np.transpose(outputs_list[i + 1]))
                i += 1


    def query(self, inputs_list):
        """опрос нейронной сети"""
        # преобразовать список входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # получение выходного сигнала
        for hn in self.w:
            inputs = self.activation_function(np.dot(hn, inputs))
        return inputs
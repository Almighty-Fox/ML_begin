import matplotlib.pyplot as plt
import numpy as np


def read_file():
    with open("kNN_data.txt", "r") as my_file:
        data_list = my_file.readlines()

    list_x, list_y, list_klass = [], [], []
    for line in data_list:
        point_x, point_y, klass = line.split()
        list_x.append(float(point_x))
        list_y.append(float(point_y))
        list_klass.append(int(klass))

    return list_x, list_y, list_klass


def grafik(list_x, list_y, list_klass):
    plt.scatter(np.array(list_x), np.array(list_y), c=list_klass)
    plt.show()


def kNN_method(test_point, count_N, list_x, list_y, list_klass):
    dist = [[], []]
    for ii in range(len(list_x)):
        dist[0].append(np.power((test_point[0] - list_x[ii])**2 + (test_point[1] - list_y[ii])**2, (1/2)))
        dist[1].append(list_klass[ii])
    dist = np.array(dist)
    dist = dist[:, dist[0].argsort()]  # сортируем двумерный массив по первой строке
    dist_N = dist[:][:count_N]  # рассматриваем count_N ближайших соседей
    max_item = lambda s: max(t := {i: np.sum(s == i) for i in s}, key=t.get)  # лямюда функция для нахождения максимально встречаемого класса
    true_klass = max_item(dist_N[1])  # находим максимально встречаемый класс

    return true_klass


if __name__ == "__main__":
    list_x, list_y, list_klass = read_file()
    # list_x = [1, 35, 6, 4, 6, 7, 8]
    # list_y = [3, 5, 6, 3, 5, 7, 4]
    # list_klass = [0, 1, 0, 1, 1, 0, 0]
    count_N = 4
    # grafik(list_x, list_y, list_klass)
    test_point = [2.4928867110688597, 3.141029945902531]
    true_klass = kNN_method(test_point, count_N, list_x, list_y, list_klass)
    print("Класс точки ", true_klass)
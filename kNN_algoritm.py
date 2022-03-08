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
        list_klass.append(float(klass))

    return list_x, list_y, list_klass


def grafik(list_x, list_y, list_klass):
    plt.scatter(np.array(list_x), np.array(list_y), c=list_klass)
    plt.show()


if __name__ == "__main__":
    list_x, list_y, list_klass = read_file()
    grafik(list_x, list_y, list_klass)
import matplotlib.pyplot as plt
import numpy as np


def read_file():  # читаем точки из файла и заполняем массивы
    with open("kNN_data_2.txt", "r") as my_file:
        data_list = my_file.readlines()

    list_x, list_y, list_klass = [], [], []
    for line in data_list:
        point_x, point_y, klass = line.split()
        list_x.append(float(point_x))
        list_y.append(float(point_y))
        list_klass.append(int(klass))

    return list_x, list_y, list_klass


def grafik(list_x, list_y, list_klass):  # строим расположение точек
    plt.figure(1)
    plt.scatter(np.array(list_x), np.array(list_y), c=list_klass)
    plt.title("Исходные точки")
    # plt.show()


def kNN_method(test_point, count_N, list_x, list_y, list_klass):  # при заданном количестве соседей определяем класс новой точки
    dist = [[], []]
    for ii in range(len(list_x)):
        dist[0].append(np.power((test_point[0] - list_x[ii])**2 + (test_point[1] - list_y[ii])**2, (1/2)))
        dist[1].append(list_klass[ii])
    dist = np.array(dist)
    dist = dist[:, dist[0].argsort()]  # сортируем двумерный массив по первой строке
    # dist_N = dist[:][:count_N]  # рассматриваем count_N ближайших соседей
    dist_N = list(map(lambda x: x[:count_N], dist))
    max_item = lambda s: max(t := {i: np.sum(s == i) for i in s}, key=t.get)  # лямбда функция для нахождения максимально встречаемого класса
    true_klass = max_item(dist_N[1])  # находим максимально встречаемый класс

    return true_klass


def kNN_optimization(list_x, list_y, list_klass):  # строим график зависимости процента угадывания классов исходных точек от количества ближайших соседей
    list_ug = np.zeros((len(list_x), len(list_x)-1))  # массив для хранения совпадения предсказания с истинным классом для каждой точки для всех количеств ближайших соседей
    for index in range(len(list_x)):  # начинаем цикл по всем точкам
        print("Обучение ", '%.1f' % ((index + 1)/len(list_x)*100), "%")  # процент обучения
        test_point = [list_x[index], list_y[index]]  # назначаем текущую точку тестовой
        list_x_opt, list_y_opt, list_klass_opt = list_x[:], list_y[:], list_klass[:]  # клонируем массивы
        list_x_opt.pop(index), list_y_opt.pop(index), list_klass_opt.pop(index)  # удаляем текущую точку из массива всех точек, что бы не считать соседом себя
        for cur_N in range(1, len(list_x)):  # начинаем массив по количству соседей
            think_klass = kNN_method(test_point, cur_N, list_x_opt, list_y_opt, list_klass_opt)  # вызываем функцию для прогнозирования класса текущей точки
            if int(think_klass) == int(list_klass[index]):   # если класс совпал и истинным, то
                list_ug[index][cur_N-1] = 1  # то в массив забиваем 1

    ver = np.array([sum(list(zip(*list_ug))[i]) for i in range(len(list_x)-1)]) / len(list_x) * 100  # массив вероятности угадывания классов всех тренировочных точек в зависимости от количества соседей
    return ver


if __name__ == "__main__":
    list_x, list_y, list_klass = read_file()
    # count_N = 4
    grafik(list_x, list_y, list_klass)
    # test_point = [2.4928867110688597, 3.141029945902531]
    # true_klass = kNN_method(test_point, count_N, list_x, list_y, list_klass)
    # print("Класс точки ", true_klass)
    ver = kNN_optimization(list_x, list_y, list_klass)
    plt.figure(2)
    plt.scatter(range(len(list_x)-1), ver)
    plt.xlabel("Количество ближайших соседей")
    plt.ylabel("Вероятность угадывания")
    plt.show()

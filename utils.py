from statistics import mean


def threshold(array):
    avg = mean(array)
    for i, val in enumerate(array):
        if val > avg:
            array[i] = 1
        else:
            array[i] = 0
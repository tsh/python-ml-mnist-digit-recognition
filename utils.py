

def threshold(array):
    for i, val in enumerate(array):
        if val > 127:
            array[i] = 1
        else:
            array[i] = 0
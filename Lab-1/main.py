import numpy as np
import random as rm


def y(x1, x2, x3):
    return (a0 + a1 * x1 + a2 * x2 + a3 * x3)
    
def calculate_x0(x_array):
    return (max(x_array) + min(x_array))/2
    
def calculate_dx(x_array, x0):
    return x0 - min(x_array)
    
def normalized_x(x, x0, dx):
    return (x - x0)/dx
    
    
if __name__ == "__main__":
    a0 = 2
    a1 = 4
    a2 = 6
    a3 = 8
    min_x = 0
    max_x = 20
    
    x1_array, x2_array, x3_array = [], [], []
    every_x_array = []
    y_array = []
    for i in range(8):
        x1 = rm.randint(min_x, max_x)
        x2 = rm.randint(min_x, max_x)
        x3 = rm.randint(min_x, max_x)
        y_value = y(x1, x2, x3)
        
        x1_array.append(x1)
        x2_array.append(x2)
        x3_array.append(x3)
        y_array.append(y_value)
        every_x_array.append([x1, x2, x3])
    
    x0_array = [calculate_x0(x1_array), calculate_x0(x2_array), calculate_x0(x3_array)]
    dx_array = [calculate_dx(x1_array, x0_array[0]), calculate_dx(x2_array, x0_array[1]), calculate_dx(x3_array, x0_array[2])]
    
    normalized_x_array = np.ones((8, 3))
    for i in range(8):
        for j in range(3):
            normalized_x_array[i][j] = normalized_x(every_x_array[i][j], x0_array[j], dx_array[j])

    average_y = sum(y_array)/len(y_array)
    difference_array = []
    for i in range(8):
        difference_array.append(y_array[i] - average_y)
    
    print("Номер | [X1, X2, X3] | Y | Різниця з Yсер | [Xn1, Xn2, Xn3]")
    for i in range(8):
        print("{}    | {}   | {}   | {}   | {}".format(i + 1, every_x_array[i], y_array[i], difference_array[i], normalized_x_array[i]))
    print("\nX0   | {} {} {}".format(x0_array[0], x0_array[1], x0_array[2]))
    print("dx   | {} {} {}".format(dx_array[0], dx_array[1], dx_array[2]))
    print("Середнє значення Y {}".format(average_y))
    
    copy_array = difference_array.copy()
    copy_array.sort()
    for i in range(8):
        if copy_array[i] > 0:
            continue
        if copy_array[i + 1] < 0:
            continue
        answer_index = difference_array.index(copy_array[i])
        print("\nТочка плану, що задовольняє критерію вибору оптимальності: {}".format(every_x_array[answer_index]))
        break

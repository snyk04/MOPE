import math
import numpy as np
from random import randint


def f_uv(u, v):
        if u >= v:
            return u / v
        else:
            return v / u


if __name__ == "__main__":
    p_list = (0.99, 0.98, 0.95, 0.90)
    rkr_table = {2: (1.73, 1.72, 1.71, 1.69),
                 6: (2.16, 2.13, 2.10, 2.00),
                 8: (2.43, 4.37, 2.27, 2.17),
                 10: (2.62, 2.54, 2.41, 2.29),
                 12: (2.75, 2.66, 2.52, 2.39),
                 15: (2.9, 2.8, 2.64, 2.49),
                 20: (3.08, 2.96, 2.78, 2.62)}

    min_y_limit, max_y_limit, m = -10, 90, 5
    x1_min, x1_min_n = 10, -1
    x1_max, x1_max_n = 40, 1
    x2_min, x2_min_n = -30, -1
    x2_max, x2_max_n = 45, 1

    y_matrix = [[randint(min_y_limit, max_y_limit) for i in range(m)] for j in range(3)]

    average_y = [sum(y_matrix[i][j] for j in range(m)) / m for i in range(3)]

    sigma_2_1 = sum([(j - average_y[0]) ** 2 for j in y_matrix[0]]) / m
    sigma_2_2 = sum([(j - average_y[1]) ** 2 for j in y_matrix[1]]) / m
    sigma_2_3 = sum([(j - average_y[2]) ** 2 for j in y_matrix[2]]) / m

    sigma_teta = math.sqrt((2 * (2 * m - 2)) / (m * (m - 4)))

    f_uv_1 = f_uv(sigma_2_1, sigma_2_2)
    f_uv_2 = f_uv(sigma_2_3, sigma_2_1)
    f_uv_3 = f_uv(sigma_2_3, sigma_2_2)

    teta_uv_1 = ((m - 2) / m) * f_uv_1
    teta_uv_2 = ((m - 2) / m) * f_uv_2
    teta_uv_3 = ((m - 2) / m) * f_uv_3

    r_uv_1 = abs(teta_uv_1 - 1) / sigma_teta
    r_uv_2 = abs(teta_uv_2 - 1) / sigma_teta
    r_uv_3 = abs(teta_uv_3 - 1) / sigma_teta

    m_x1 = (-1 + 1 - 1) / 3
    m_x2 = (-1 - 1 + 1) / 3
    m_y = sum(average_y) / 3
    a_1 = (1 + 1 + 1) / 3
    a_2 = (1 - 1 - 1) / 3
    a_3 = (1 + 1 + 1) / 3
    a_11 = (-1 * average_y[0] + 1 * average_y[1] - 1 * average_y[2]) / 3
    a_22 = (-1 * average_y[0] - 1 * average_y[1] + 1 * average_y[2]) / 3

    b_0 = np.linalg.det(np.dot([[m_y, m_x1, m_x2],
                               [a_11, a_1, a_2],
                               [a_22, a_2, a_3]],
                              np.linalg.inv([[1, m_x1, m_x2],
                                             [m_x1, a_1, a_2],
                                             [m_x2, a_2, a_3]])))

    b_1 = np.linalg.det(np.dot([[1, m_y, m_x2],
                               [m_x1, a_11, a_2],
                               [m_x2, a_22, a_3]],
                              np.linalg.inv([[1, m_x1, m_x2],
                                             [m_x1, a_1, a_2],
                                             [m_x2, a_2, a_3]])))

    b_2 = np.linalg.det(np.dot([[1, m_x1, m_y],
                               [m_x1, a_1, a_11],
                               [m_x2, a_2, a_22]],
                              np.linalg.inv([[1, m_x1, m_x2],
                                             [m_x1, a_1, a_2],
                                             [m_x2, a_2, a_3]])))

    normalized_y = b_0 - b_1 + b_2

    dx_1 = math.fabs(x1_max - x1_min) / 2
    dx_2 = math.fabs(x2_max - x2_min) / 2
    x_10 = (x1_max + x1_min) / 2
    x_20 = (x2_max + x2_min) / 2

    aa_0 = b_0 - b_1 * x_10 / dx_1 - b_2 * x_20 / dx_2
    aa_1 = b_1 / dx_1
    aa_2 = b_2 / dx_2

    for i in range(3):
        print("Y{}: {}, Середній Y: {}".format(i + 1, y_matrix[i], average_y[i]))
    print("\nσ² y1: {}".format(round(sigma_2_1, 4)))
    print("σ² y2: {}".format(round(sigma_2_2, 4)))
    print("σ² y3: {}".format(round(sigma_2_2, 4)))
    print("σθ = {}".format(round(sigma_teta, 4)))
    print("\nFuv1 = {}".format(round(f_uv_1, 4)))
    print("Fuv2 = {}".format(round(f_uv_2, 4)))

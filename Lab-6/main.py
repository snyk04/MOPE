import math
import numpy
import random
from _decimal import Decimal
from functools import reduce
from itertools import compress
from scipy.stats import f, t


# Ініціалізація змінних
x_min = [10, -30, -30]
x_max = [40, 45, -10]
x0 = [(x_max[_] + x_min[_]) / 2 for _ in range(3)]
dx = [x_max[_] - x0[_] for _ in range(3)]

amount_of_coefficients = 0

norm_plan_raw = [[-1, -1, -1],
                 [-1, +1, +1],
                 [+1, -1, +1],
                 [+1, +1, -1],
                 [-1, -1, +1],
                 [-1, +1, -1],
                 [+1, -1, -1],
                 [+1, +1, +1],
                 [-1.73, 0, 0],
                 [+1.73, 0, 0],
                 [0, -1.73, 0],
                 [0, +1.73, 0],
                 [0, 0, -1.73],
                 [0, 0, +1.73]]

natural_plan_raw = [[x_min[0], x_min[1], x_min[2]],
                    [x_min[0], x_min[1], x_max[2]],
                    [x_min[0], x_max[1], x_min[2]],
                    [x_min[0], x_max[1], x_max[2]],
                    [x_max[0], x_min[1], x_min[2]],
                    [x_max[0], x_min[1], x_max[2]],
                    [x_max[0], x_max[1], x_min[2]],
                    [x_max[0], x_max[1], x_max[2]],
                    [-1.73*dx[0]+x0[0], x0[1],             x0[2]],
                    [1.73*dx[0]+x0[0],  x0[1],             x0[2]],
                    [x0[0],             -1.73*dx[1]+x0[1], x0[2]],
                    [x0[0],             1.73*dx[1]+x0[1],  x0[2]],
                    [x0[0],             x0[1],             -1.73*dx[2]+x0[2]],
                    [x0[0],             x0[1],             1.73*dx[2]+x0[2]],
                    [x0[0],             x0[1],             x0[2]]]


# Основні функції
def regression_equation(x1, x2, x3, coefficients, importance=[True] * 11):
    factors_array = [1, x1, x2, x3, x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3]
    return sum([el[0] * el[1] for el in compress(zip(coefficients, factors_array), importance)])


def func(x1, x2, x3):
    coefficients = [5.9, 4.0, 3.5, 8.2, 4.3, 0.7, 9.3, 6.1, 0.5, 8.6, 3.1]
    return regression_equation(x1, x2, x3, coefficients)


def generate_factors_table(raw_array):
    raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] + list(
        map(lambda x: x ** 2, row)) for row in raw_array]
    return list(map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))


def generate_y(m, factors_table):
    return [[round(func(row[0], row[1], row[2]) + random.randint(-5, 5), 3) for _ in range(m)] for row in factors_table]


# Вивід результатів
def print_matrix(m, n, factors, y_values, additional_text=":"):
    labels_table = list(map(lambda x: x.ljust(10),
                            ["x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"] + [
                                "y{}".format(i + 1) for i in range(m)]))
    rows_table = [list(factors[i]) + list(y_values[i]) for i in range(n)]
    print("\nМатриця планування" + additional_text)
    print(" ".join(labels_table))
    print("\n".join([" ".join(map(lambda j: "{:<+10}".format(j), rows_table[i])) for i in range(len(rows_table))]))
    print("\t")


def print_equation(coefficients, importance=[True] * 11):
    x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    coefficients_to_print = list(compress(coefficients, importance))
    equation = " ".join(
        ["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), coefficients_to_print)), x_i_names)])
    print("Рівняння регресії: y = " + equation)


def set_factors_table(factors_table):
    def x_i(i):
        with_null_factor = list(map(lambda x: [1] + x, generate_factors_table(factors_table)))
        res = [row[i] for row in with_null_factor]
        return numpy.array(res)

    return x_i


def m_ij(*arrays):
    return numpy.average(reduce(lambda accum, el: accum * el, list(map(lambda el: numpy.array(el), arrays))))


def find_coefficients(factors, y_values):
    x_i = set_factors_table(factors)
    coefficients = [[m_ij(x_i(column), x_i(row)) for column in range(11)] for row in range(11)]
    y_numpy = list(map(lambda row: numpy.average(row), y_values))
    free_values = [m_ij(y_numpy, x_i(i)) for i in range(11)]
    beta_coefficients = numpy.linalg.solve(coefficients, free_values)
    return list(beta_coefficients)


# Критерії
def cochran_criteria(m, n, y_table):
    def get_cochran_value(f1, f2, q):
        part_result1 = q / f2
        params = [part_result1, f1, (f2 - 1) * f1]
        fisher = f.isf(*params)
        result = fisher / (fisher + (f2 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    print("Перевірка рівномірності дисперсій за критерієм Кохрена: m = {}, N = {}".format(m, n))
    y_variations = [numpy.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation / sum(y_variations)
    f1 = m - 1
    f2 = n
    p = 0.95
    q = 1 - p
    gt = get_cochran_value(f1, f2, q)
    print("Gp = {}; Gt = {}; f1 = {}; f2 = {}; q = {:.2f}".format(gp, gt, f1, f2, q))
    if gp < gt:
        print("Gp < Gt => дисперсії рівномірні - все правильно")
        return True
    else:
        print("Gp > Gt => дисперсії нерівномірні - треба ще експериментів")
        return False


def student_criteria(m, n, y_table, beta_coefficients):
    global amount_of_coefficients

    def get_student_value(f3, q):
        return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()

    print("\nПеревірка значимості коефіцієнтів регресії за критерієм Стьюдента: m = {}, N = {} ".format(m, n))
    average_variation = numpy.average(list(map(numpy.var, y_table)))
    variation_beta_s = average_variation / n / m
    standard_deviation_beta_s = math.sqrt(variation_beta_s)
    t_i = [abs(beta_coefficients[i]) / standard_deviation_beta_s for i in range(len(beta_coefficients))]
    f3 = (m - 1) * n
    q = 0.05
    t_our = get_student_value(f3, q)
    importance = [True if el > t_our else False for el in list(t_i)]

    amount_of_coefficients += sum(importance)

    # print result data
    print("Оцінки коефіцієнтів βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
    print("Коефіцієнти ts: " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    print("f3 = {}; q = {}; tтабл = {}".format(f3, q, t_our))
    beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
    importance_to_print = ["важливий" if i else "неважливий" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
    print(*to_print, sep="; ")
    print_equation(beta_coefficients, importance)
    return importance


def fisher_criteria(m, N, d, x_table, y_table, b_coefficients, importance):
    def get_fisher_value(f3, f4, q):
        return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()

    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05
    theoretical_y = numpy.array([regression_equation(row[0], row[1], row[2], b_coefficients) for row in x_table])
    average_y = numpy.array(list(map(lambda el: numpy.average(el), y_table)))
    s_ad = m / (N - d) * sum((theoretical_y - average_y) ** 2)
    y_variations = numpy.array(list(map(numpy.var, y_table)))
    s_v = numpy.average(y_variations)
    f_p = float(s_ad / s_v)
    f_t = get_fisher_value(f3, f4, q)
    theoretical_values_to_print = list(
        zip(map(lambda x: "x1 = {0[1]:<10} x2 = {0[2]:<10} x3 = {0[3]:<10}".format(x), x_table), theoretical_y))
    print("\nПеревірка адекватності моделі за критерієм Фішера: m = {}, N = {} для таблиці y_table".format(m, N))
    print("Теоретичні значення y для різних комбінацій факторів:")
    print("\n".join(["{arr[0]}: y = {arr[1]}".format(arr=el) for el in theoretical_values_to_print]))
    print("Fp = {}, Ft = {}".format(f_p, f_t))
    print("Fp < Ft => модель адекватна" if f_p < f_t else "Fp > Ft => модель неадекватна")
    return True if f_p < f_t else False


def main():
    m = 3
    n = 15

    amount_of_iterations = 0

    while amount_of_coefficients < 50:
        natural_plan = generate_factors_table(natural_plan_raw)
        y_arr = generate_y(m, natural_plan_raw)
        while not cochran_criteria(m, n, y_arr):
            m += 1
            y_arr = generate_y(m, natural_plan)

        print_matrix(m, n, natural_plan, y_arr, " для натуралізованих факторів:")
        coefficients = find_coefficients(natural_plan, y_arr)
        print_equation(coefficients)
        importance = student_criteria(m, n, y_arr, coefficients)
        d = len(list(filter(None, importance)))
        fisher_criteria(m, n, d, natural_plan, y_arr, coefficients, importance)
        
        if amount_of_coefficients < 50:
            amount_of_iterations += 1

    print(f"\nКількість ітерацій: {amount_of_iterations}")


if __name__ == "__main__":
    main()

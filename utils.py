"""
ARKs ML Utils
"""
import numpy as np


def random_parameter_generation(ranges, is_exp, num_combinations):
    """
    generates parameter combinations for random parameter search
    :param ranges:
    :param is_exp: will output 10^param instead of param
    :param num_combinations:
    :return: list of lists, each caching values of one PARAMETER
    """
    lists = list()
    for (i, (min_b, max_b)) in enumerate(ranges):
        sublist = list()
        for _ in range(num_combinations):
            rnd = np.random.uniform() * (max_b - min_b) + min_b
            if is_exp[i] is True:
                rnd = np.power(10., rnd)
            sublist.append(rnd)
        lists.append(sublist)
    return lists


def modified_levenshtein_distance(s1, s2):
    """
    cost of each operation is now weighted with its position in the string
    :param s1:
    :param s2:
    :return:
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    maxlen = max(len(s1), len(s2))

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1.]
        for i1, c1 in enumerate(s1):
            cost = (maxlen - min(i1, i2))/maxlen
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(cost + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]



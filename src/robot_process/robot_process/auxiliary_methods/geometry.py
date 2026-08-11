"""采样式路径规划使用的基础几何函数。"""

from itertools import tee

import numpy as np


def dist_between_points(a, b):
    """返回两点 ``a``、``b`` 之间的欧氏距离。"""
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance


def pairwise(iterable):
    """把序列转换为相邻元素对：``(s0,s1), (s1,s2), ...``。"""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def es_points_along_line(start, end, r):
    """沿线段生成间距不大于 ``r`` 的等距采样点。

    当计算出的采样位置少于两个时不产生点；否则结果包含首尾点。
    """
    d = dist_between_points(start, end)
    n_points = int(np.ceil(d / r))
    if n_points > 1:
        step = d / (n_points - 1)
        for i in range(n_points):
            next_point = steer(start, end, i * step)
            yield next_point


def steer(start, goal, d):
    """从 ``start`` 朝 ``goal`` 方向前进距离 ``d``，返回得到的点。"""
    start, end = np.array(start), np.array(goal)
    v = end - start
    u = v / (np.sqrt(np.sum(v ** 2)))
    steered_point = start + u * d
    return tuple(steered_point)

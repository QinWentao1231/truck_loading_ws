"""RRT/RRT* 的欧氏距离代价函数。"""

from auxiliary_methods.geometry import dist_between_points


def cost_to_go(a: tuple, b: tuple) -> float:
    """用两点欧氏距离估计从 ``a`` 到 ``b`` 的剩余代价。"""
    return dist_between_points(a, b)


def path_cost(E, a, b):
    """沿父边映射 ``E[child] = parent`` 累加从 ``a`` 到 ``b`` 的路径长度。"""
    cost = 0
    while not b == a:
        p = E[b]
        cost += dist_between_points(b, p)
        b = p

    return cost


def segment_cost(a, b):
    """返回线段 ``a→b`` 的欧氏长度。"""
    return dist_between_points(a, b)

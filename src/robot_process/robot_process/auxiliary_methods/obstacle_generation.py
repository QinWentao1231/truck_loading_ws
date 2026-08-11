"""搜索空间的随机轴对齐障碍物及 R-tree 数据生成器。"""

import random
import uuid

import numpy as np


def generate_random_obstacles(X, start, end, n):
    """生成 ``n`` 个互不相交且不覆盖起终点的随机轴对齐障碍物。

    本函数只检查新障碍物与既有障碍物、起点和终点的几何关系，不验证剩余
    自由空间是否仍然连通；调用方仍需通过路径规划结果判断起终点是否可达。
    """
    # 当前仅生成轴对齐超矩形；edge_lengths 表示从中心到各边的半尺寸。
    i = 0
    obstacles = []
    while i < n:
        center = np.empty(len(X.dimension_lengths), np.float64)
        scollision = True
        fcollision = True
        edge_lengths = []
        for j in range(X.dimensions):
            # 每个方向的半尺寸限制在该维搜索跨度的 1%～10%。
            max_edge_length = (X.dimension_lengths[j][1] - X.dimension_lengths[j][0]) / 10.0
            min_edge_length = (X.dimension_lengths[j][1] - X.dimension_lengths[j][0]) / 100.0
            edge_length = random.uniform(min_edge_length, max_edge_length)
            center[j] = random.uniform(X.dimension_lengths[j][0] + edge_length,
                                       X.dimension_lengths[j][1] - edge_length)
            edge_lengths.append(edge_length)

            if abs(start[j] - center[j]) > edge_length:
                scollision = False
            if abs(end[j] - center[j]) > edge_length:
                fcollision = False

        # 组装 R-tree 使用的 (min..., max...) 边界。
        min_corner = np.empty(X.dimensions, np.float64)
        max_corner = np.empty(X.dimensions, np.float64)
        for j in range(X.dimensions):
            min_corner[j] = center[j] - edge_lengths[j]
            max_corner[j] = center[j] + edge_lengths[j]
        obstacle = np.append(min_corner, max_corner)
        # 与已有障碍相交，或覆盖起点/终点时重新采样。
        if len(list(X.obs.intersection(obstacle))) > 0 or scollision or fcollision:
            continue
        i += 1
        obstacles.append(obstacle)
        X.obs.add(uuid.uuid4(), tuple(obstacle), tuple(obstacle))

    return obstacles


def obstacle_generator(obstacles):
    """把障碍物列表转换为 R-tree 批量构造所需的三元组迭代器。"""
    for i, obstacle in enumerate(obstacles):
        yield (i, obstacle, obstacle)

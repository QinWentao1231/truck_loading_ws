"""基于 R-tree 的轴对齐搜索空间与离散碰撞查询。"""

import numpy as np
from rtree import index

from auxiliary_methods.geometry import es_points_along_line
from auxiliary_methods.obstacle_generation import obstacle_generator


class SearchSpace(object):
    """保存各维边界及障碍物索引，供路径点和采样线段检查。"""

    def __init__(self, dimension_lengths, O=None):
        """用各维 ``[min, max]`` 范围和障碍物列表初始化索引。

        当前实现会直接对 ``O`` 取长度，因此无障碍时也应显式传入空列表。
        """
        # 搜索空间至少二维，且每维用 [min, max] 表示。
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] > i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths
        p = index.Property()
        p.dimension = self.dimensions
        if len(O) == 0:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # R-tree 中每个障碍物为 (min..., max...) 的轴对齐包围盒。
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)

    def obstacle_free(self, x, size=None):
        """检查点或 AABB 的八个角点是否落入障碍物。

        ``x`` 是 AABB 最小角，``size`` 是三轴尺寸。该快速检查不验证障碍物
        完全穿过包围盒内部但未包含任一角点的特殊相交情况。
        """
        if size is None:
            size = (0, 0, 0)
        x_list = [x, (x[0] + size[0], x[1], x[2]),
                  (x[0] + size[0], x[1] + size[1], x[2]), (x[0], x[1] + size[1], x[2]),
                  (x[0], x[1], x[2] + size[2]), (x[0] + size[0], x[1], x[2] + size[2]),
                  (x[0] + size[0], x[1] + size[1], x[2] + size[2]), (x[0], x[1] + size[1], x[2] + size[2])]
        for i in x_list:
            if self.obs.count(i) != 0:
                return False
        return True

    def collision_free(self, start, end, r, size):
        """离散采样移动 AABB 的八条角点轨迹，判断线段是否无碰撞。"""
        starts = [start,
                  (start[0] + size[0], start[1], start[2]),
                  (start[0] + size[0], start[1] + size[1], start[2]),
                  (start[0], start[1] + size[1], start[2]),
                  (start[0], start[1], start[2] + size[2]),
                  (start[0] + size[0], start[1], start[2] + size[2]),
                  (start[0] + size[0], start[1] + size[1], start[2] + size[2]),
                  (start[0], start[1] + size[1], start[2] + size[2])]
        ends = [end,
                (end[0] + size[0], end[1], end[2]),
                (end[0] + size[0], end[1] + size[1], end[2]),
                (end[0], end[1] + size[1], end[2]),
                (end[0], end[1], end[2] + size[2]),
                (end[0] + size[0], end[1], end[2] + size[2]),
                (end[0] + size[0], end[1] + size[1], end[2] + size[2]),
                (end[0], end[1] + size[1], end[2] + size[2])]
        for s, e in zip(starts, ends):
            points = es_points_along_line(s, e, r)
            coll_free = all(map(self.obstacle_free, points))
            if not coll_free:
                return False
        return True

    def sample(self):
        """在各维边界内均匀采样一点，不保证该点无碰撞。"""
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)

    def sample_free(self, size):
        """持续采样，直到得到移动 AABB 角点不在障碍物内的位置。"""
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x, size):
                return x

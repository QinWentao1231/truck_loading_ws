"""RRT 与 RRT* 共用的树维护、采样和路径重建逻辑。"""

import random
import numpy as np
from auxiliary_methods.tree import Tree
from auxiliary_methods.geometry import steer


class RRTBase(object):
    """单树 RRT 基类；搜索空间负责实际点/线段碰撞判断。"""
    def __init__(self, X, Q, x_init, x_goal, size, max_samples, r, prc=0.3):
        """保存搜索空间、扩展参数、移动 AABB 尺寸和目标检查策略。"""
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.Q = Q
        self.r = r
        self.prc = prc
        self.x_init = x_init
        self.x_goal = x_goal
        self.size = size
        self.trees = []  # 当前实现只创建一棵树，列表结构保留扩展能力。
        self.add_tree()

    def add_tree(self):
        """创建一棵与搜索空间维数一致的空树。"""
        self.trees.append(Tree(self.X))

    def add_vertex(self, tree, v):
        """向指定树的 R-tree 索引插入顶点并更新计数。"""
        self.trees[tree].V.insert(0, v + v, v)
        self.trees[tree].V_count += 1
        self.samples_taken += 1

    def add_edge(self, tree, child, parent):
        """记录指定树的一条父边 ``E[child] = parent``。"""
        self.trees[tree].E[child] = parent

    def nearby(self, tree, x, n):
        """返回指定点附近最多 ``n`` 个顶点的 R-tree 迭代器。"""
        return self.trees[tree].V.nearest(x, num_results=n, objects="raw")

    def get_nearest(self, tree, x):
        """返回指定树中离 ``x`` 最近的顶点。"""
        return next(self.nearby(tree, x, 1))

    def new_and_near(self, tree, q):
        """随机采样自由点，按 ``q[0]`` 步长生成新点及其最近树顶点。"""
        x_rand = self.X.sample_free(self.size)
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
        # 新点必须未入树，且其移动包围盒角点不落入障碍物。
        if not self.trees[0].V.count(x_new) == 0 or not self.X.obstacle_free(x_new, self.size):
            return None, None
        self.samples_taken += 1
        return x_new, x_nearest

    def connect_to_point(self, tree, x_a, x_b):
        """若 ``x_b`` 未入树且 ``x_a→x_b`` 无碰撞，则添加顶点和父边。"""
        if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r, self.size):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    def can_connect_to_goal(self, tree):
        """检查指定树的最近顶点是否可以无碰撞连接目标点。"""
        x_nearest = self.get_nearest(tree, self.x_goal)
        if self.x_goal in self.trees[tree].E and x_nearest in self.trees[tree].E[self.x_goal]:
            # 目标已经通过当前最近顶点挂入父边映射。
            return True
        if self.X.collision_free(x_nearest, self.x_goal, self.r, self.size):
            return True
        return False

    def get_path(self):
        """目标可连接时补上目标父边并返回路径，否则返回 ``None``。"""
        if self.can_connect_to_goal(0):
            self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        return None

    def connect_to_goal(self, tree):
        """不做碰撞复核，直接把目标父节点设为当前最近顶点。"""
        x_nearest = self.get_nearest(tree, self.x_goal)
        self.trees[tree].E[self.x_goal] = x_nearest

    def reconstruct_path(self, tree, x_init, x_goal):
        """沿父边回溯路径；结果不重复包含 ``x_init``，但包含 ``x_goal``。"""
        path = [x_goal]
        current = x_goal
        if x_init == x_goal:
            return path
        while not self.trees[tree].E[current] == x_init:
            path.append(self.trees[tree].E[current])
            current = self.trees[tree].E[current]
        path.reverse()
        return path

    def check_solution(self):
        """按 ``prc`` 概率提前检查目标连接，达到采样上限时强制检查。"""
        if self.prc and random.random() < self.prc:
            path = self.get_path()
            if path is not None:
                return True, path
        # check if can connect to goal after generating max_samples
        if self.samples_taken >= self.max_samples:
            return True, self.get_path()
        return False, None

    def bound_point(self, point):
        """把每个坐标分量裁剪到搜索空间边界。"""
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)

"""带邻域选父与重连步骤的 RRT* 实现。"""

from operator import itemgetter

from auxiliary_methods.heuristics import cost_to_go
from auxiliary_methods.heuristics import segment_cost, path_cost
from rrt_planning.rrt import RRT


class RRTStar(RRT):
    """在 RRT 搜索基础上按累计路径代价选择和重连邻点。"""
    def __init__(self, X, Q, x_init, x_goal, size, max_samples, r, prc=0.01, rewire_count=None):
        """初始化 RRT*；``rewire_count`` 为每次考虑的邻点数。

        当前实现会把 ``None`` 转换为0，因此省略该参数时不执行邻域选父和重连。
        """
        super().__init__(X, Q, x_init, x_goal, size, max_samples, r, prc)
        self.rewire_count = rewire_count if rewire_count is not None else 0
        self.c_best = float('inf')  # 当前已知最优路径长度；外部可更新该值用于剪枝。

    def get_nearby_vertices(self, tree, x_init, x_new):
        """返回新点附近的 ``(经该邻点到新点的总代价, 邻点)`` 升序列表。"""
        X_near = self.nearby(tree, x_new, self.current_rewire_count(tree))
        L_near = [(path_cost(self.trees[tree].E, x_init, x_near) + segment_cost(x_near, x_new), x_near) for
                  x_near in X_near]
        L_near.sort(key=itemgetter(0))

        return L_near

    def rewire(self, tree, x_new, L_near):
        """若经 ``x_new`` 可缩短且连线无碰撞，则改写邻点的父节点。"""
        for c_near, x_near in L_near:
            curr_cost = path_cost(self.trees[tree].E, self.x_init, x_near)
            tent_cost = path_cost(self.trees[tree].E, self.x_init, x_new) + segment_cost(x_new, x_near)
            if tent_cost < curr_cost and self.X.collision_free(x_near, x_new, self.r, self.size):
                self.trees[tree].E[x_near] = x_new

    def connect_shortest_valid(self, tree, x_new, L_near):
        """按总代价从小到大尝试，把新点连接到首个无碰撞且未被剪枝的邻点。"""
        for c_near, x_near in L_near:
            if c_near + cost_to_go(x_near, self.x_goal) < self.c_best and self.connect_to_point(tree, x_near, x_new):
                break

    def current_rewire_count(self, tree):
        """返回本次可参与重连的邻点数，不超过树的现有顶点数。"""
        if self.rewire_count is None:
            return self.trees[tree].V_count

        return min(self.trees[tree].V_count, self.rewire_count)

    def rrt_star(self):
        """执行 RRT* 搜索并返回路径点列表；无法连接目标时返回 ``None``。"""
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            for q in self.Q:  # 依次使用各组扩展步长。
                for i in range(q[1]):  # 在该步长下尝试指定次数。
                    x_new, x_nearest = self.new_and_near(0, q)
                    if x_new is None:
                        continue

                    # 计算附近顶点作为候选父节点时的累计代价。
                    L_near = self.get_nearby_vertices(0, self.x_init, x_new)

                    # 连接到总代价最小的可达候选父节点。
                    self.connect_shortest_valid(0, x_new, L_near)

                    if x_new in self.trees[0].E:
                        # 用新点尝试缩短附近顶点的已有路径。
                        self.rewire(0, x_new, L_near)

                    solution = self.check_solution()
                    if solution[0]:
                        return solution[1]

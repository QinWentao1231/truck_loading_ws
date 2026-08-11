"""基础快速扩展随机树（RRT）实现。"""

from rrt_planning.rrt_base import RRTBase


class RRT(RRTBase):
    """持续扩展随机树，直到连接目标或达到采样上限。"""
    def __init__(self, X, Q, x_init, x_goal, size, max_samples, r, prc=0.01):
        """初始化 RRT 参数；``Q`` 中每项为 ``(扩展步长, 尝试次数)``。"""
        super().__init__(X, Q, x_init, x_goal, size, max_samples, r, prc)

    def rrt_search(self):
        """扩展树并返回路径点列表；采样耗尽且无法连接时返回 ``None``。"""
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            for q in self.Q:  # 依次使用各组扩展步长。
                for i in range(q[1]):  # 在该步长下尝试指定次数。
                    x_new, x_nearest = self.new_and_near(0, q)

                    if x_new is None:
                        continue

                    # 将新采样点连接到最近的可达顶点。
                    self.connect_to_point(0, x_nearest, x_new)

                    solution = self.check_solution()
                    if solution[0]:
                        return solution[1]

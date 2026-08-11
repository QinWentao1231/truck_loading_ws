"""RRT 使用的顶点空间索引和父子边容器。"""

from rtree import index


class Tree(object):
    """用 R-tree 存顶点，并以 ``E[child] = parent`` 保存有向边。"""

    def __init__(self, X):
        """按搜索空间维数初始化空树。"""
        p = index.Property()
        p.dimension = X.dimensions
        self.V = index.Index(interleaved=True, properties=p)  # R-tree 顶点索引
        self.V_count = 0
        self.E = {}  # 父边映射：E[child] = parent

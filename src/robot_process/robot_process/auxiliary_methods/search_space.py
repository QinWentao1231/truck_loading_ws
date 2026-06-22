import numpy as np
from rtree import index

from auxiliary_methods.geometry import es_points_along_line
from auxiliary_methods.obstacle_generation import obstacle_generator


class SearchSpace(object):

    def __init__(self, dimension_lengths, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] > i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions
        # if O is None:
        if len(O) == 0:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)

    def obstacle_free(self, x, size=None):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
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
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
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
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)

    def sample_free(self, size):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x, size):
                return x


import os
import plotly as py
from plotly import graph_objs as go

colors = ['darkblue', 'teal']


def _resolve_log_dir(file_path):
    """
    从文件路径推导 <ws_root>/log/<pkg_name>/，兼容两种启动方式：
      - VSCode 直接运行：<ws>/src/<pkg>/<pkg>/...
      - ros2 run 安装路径：<ws>/install/<pkg>/lib/<pkg>/...
    通过查找路径中的 src 或 install 段定位工作空间根，不依赖固定层级。
    """
    real = os.path.realpath(os.path.abspath(file_path))
    parts = real.split(os.sep)
    for i, part in enumerate(parts):
        if part in ('src', 'install'):
            ws_root = os.sep.join(parts[:i]) or os.sep
            pkg_name = parts[i + 1] if i + 1 < len(parts) else 'robot_process'
            return os.path.join(ws_root, 'log', pkg_name)
    # 兜底：文件所在目录上4级
    return os.path.join(os.path.dirname(real), '..', '..', '..', '..', 'log')


class Plot(object):
    def __init__(self, filename):
        """
        Create a plot
        :param filename: filename (without extension)
        Saved to: <ws_root>/log/<pkg_name>/<filename>.html
        """
        log_dir = _resolve_log_dir(__file__)
        os.makedirs(log_dir, exist_ok=True)
        self.filename = os.path.join(log_dir, filename + ".html")
        self.data = []
        self.layout = {'title': 'Plot',
                       'showlegend': False
                       }

        self.fig = {'data': self.data,
                    'layout': self.layout}

    def plot_tree(self, X, trees):
        """
        Plot tree
        :param X: Search Space
        :param trees: list of trees
        """
        if X.dimensions == 2:  # plot in 2D
            self.plot_tree_2d(trees)
        elif X.dimensions == 3:  # plot in 3D
            self.plot_tree_3d(trees)
        else:  # can't plot in higher dimensions
            print("Cannot plot in > 3 dimensions")

    def plot_tree_2d(self, trees):
        """
        Plot 2D trees
        :param trees: trees to plot
        """
        for i, tree in enumerate(trees):
            for start, end in tree.E.items():
                if end is not None:
                    trace = go.Scatter(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        line=dict(
                            color=colors[i]
                        ),
                        mode="lines"
                    )
                    self.data.append(trace)

    def plot_tree_3d(self, trees):
        """
        Plot 3D trees
        :param trees: trees to plot
        """
        for i, tree in enumerate(trees):
            for start, end in tree.E.items():
                if end is not None:
                    trace = go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        line=dict(
                            color=colors[i]
                        ),
                        mode="lines"
                    )
                    self.data.append(trace)

    def plot_obstacles(self, X, O):
        """
        Plot obstacles
        :param X: Search Space
        :param O: list of obstacles
        """
        if X.dimensions == 2:  # plot in 2D
            self.layout['shapes'] = []
            for O_i in O:
                # noinspection PyUnresolvedReferences
                self.layout['shapes'].append(
                    {
                        'type': 'rect',
                        'x0': O_i[0],
                        'y0': O_i[1],
                        'x1': O_i[2],
                        'y1': O_i[3],
                        'line': {
                            'color': 'purple',
                            'width': 4,
                        },
                        'fillcolor': 'purple',
                        'opacity': 0.70
                    },
                )
        elif X.dimensions == 3:  # plot in 3D
            for O_i in O:
                obs = go.Mesh3d(
                    x=[O_i[0], O_i[0], O_i[3], O_i[3], O_i[0], O_i[0], O_i[3], O_i[3]],
                    y=[O_i[1], O_i[4], O_i[4], O_i[1], O_i[1], O_i[4], O_i[4], O_i[1]],
                    z=[O_i[2], O_i[2], O_i[2], O_i[2], O_i[5], O_i[5], O_i[5], O_i[5]],
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    color='purple',
                    opacity=0.70
                )
                self.data.append(obs)
        else:  # can't plot in higher dimensions
            print("Cannot plot in > 3 dimensions")

    def plot_path(self, X, path, size, colors=None):
        """
        Plot path through Search Space
        :param X: Search Space
        :param path: path through space given as a sequence of points
        :param colors: 可选，长度等于 path 的颜色列表，为每个路径点的包围盒指定颜色；
                       不足部分用 'yellow' 补齐
        """
        if X.dimensions == 2:  # plot in 2D
            x, y = [], []
            for i in path:
                x.append(i[0])
                y.append(i[1])
            trace = go.Scatter(
                x=x,
                y=y,
                line=dict(
                    color="red",
                    width=4
                ),
                mode="lines"
            )

            self.data.append(trace)
        elif X.dimensions == 3:  # plot in 3D
            x, y, z = [], [], []
            for idx, i in enumerate(path):
                x.append(i[0])
                y.append(i[1])
                z.append(i[2])
                pt_color = colors[idx] if colors and idx < len(colors) else 'yellow'
                path_obs = go.Mesh3d(
                    x=[i[0], i[0], i[0]+size[0], i[0]+size[0], i[0], i[0], i[0]+size[0], i[0]+size[0]],
                    y=[i[1], i[1]+size[1], i[1]+size[1], i[1], i[1], i[1]+size[1], i[1]+size[1], i[1]],
                    z=[i[2], i[2], i[2], i[2], i[2]+size[2], i[2]+size[2], i[2]+size[2], i[2]+size[2]],
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    color=pt_color,
                    opacity=0.70
                )
                self.data.append(path_obs)
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                line=dict(
                    color="red",
                    width=4
                ),
                mode="lines"
            )
            self.data.append(trace)
        else:  # can't plot in higher dimensions
            print("Cannot plot in > 3 dimensions")

    def plot_start(self, X, x_init):
        """
        Plot starting point
        :param X: Search Space
        :param x_init: starting location
        """
        if X.dimensions == 2:  # plot in 2D
            trace = go.Scatter(
                x=[x_init[0]],
                y=[x_init[1]],
                line=dict(
                    color="orange",
                    width=10
                ),
                mode="markers"
            )

            self.data.append(trace)
        elif X.dimensions == 3:  # plot in 3D
            trace = go.Scatter3d(
                x=[x_init[0]],
                y=[x_init[1]],
                z=[x_init[2]],
                line=dict(
                    color="orange",
                    width=10
                ),
                mode="markers"
            )

            self.data.append(trace)
        else:  # can't plot in higher dimensions
            print("Cannot plot in > 3 dimensions")

    def plot_goal(self, X, x_goal):
        """
        Plot goal point
        :param X: Search Space
        :param x_goal: goal location
        """
        if X.dimensions == 2:  # plot in 2D
            trace = go.Scatter(
                x=[x_goal[0]],
                y=[x_goal[1]],
                line=dict(
                    color="green",
                    width=10
                ),
                mode="markers"
            )

            self.data.append(trace)
        elif X.dimensions == 3:  # plot in 3D
            trace = go.Scatter3d(
                x=[x_goal[0]],
                y=[x_goal[1]],
                z=[x_goal[2]],
                line=dict(
                    color="green",
                    width=10
                ),
                mode="markers"
            )

            self.data.append(trace)
        else:  # can't plot in higher dimensions
            print("Cannot plot in > 3 dimensions")

    def draw(self, auto_open=False):
        """
        Render the plot to a file
        """
        py.offline.plot(self.fig, filename=self.filename, auto_open=auto_open)

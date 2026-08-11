"""路径 HTML 与整面垛型 PNG 可视化工具。

逐抓 HTML 使用 Plotly 绘制路径点包围盒和已有障碍；面级 PNG 使用 Matplotlib
绘制 Y-Z 正视图及三维箱体布局。所有几何尺寸沿用主流程的毫米单位。
"""

import os
import datetime

try:
    import plotly as py
    from plotly import graph_objs as go
except ImportError:  # PNG 面级图不依赖 plotly；逐抓 HTML 由调用方记录失败并继续。
    py = None
    go = None

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


def resolve_log_dir():
    """返回 robot_process 统一日志目录。"""
    log_dir = os.path.realpath(_resolve_log_dir(__file__))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def resolve_output_dir(date_text=None):
    """返回按日期归档的可视化/检查输出目录。"""
    date_text = date_text or datetime.datetime.now().strftime('%Y%m%d')
    output_dir = os.path.join(resolve_log_dir(), str(date_text))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _action_boxes(action):
    """将一个放置动作展开为真实尺寸箱体 (x, y, z, L, W, H)。"""
    pos = action['pos']
    length, width, height = action['size']
    boxes = []
    if action['area'] == 'p1':
        y = pos[1]
        gaps = action.get('gaps', [])
        for seg_idx, count in enumerate(action['num']):
            for index in range(count):
                boxes.append((pos[0], y + index * width, pos[2],
                              length, width, height))
            y += count * width
            if seg_idx < len(gaps):
                y += gaps[seg_idx]
    elif action['area'] == 'p2':
        for index in range(sum(action['num'])):
            boxes.append((pos[0] + index * length, pos[1], pos[2],
                          length, width, height))
    elif action['area'] == 'p3':
        for index in range(sum(action['num'])):
            boxes.append((pos[0], pos[1] + index * height, pos[2],
                          length, height, width))
    return boxes


def save_face_layout(
        actions, car_size, filename, title=None, issue_ids=None,
        output_dir=None):
    """保存一整面垛型的 PNG 正视图和三维图，返回输出路径。

    matplotlib 采用延迟导入；现场环境缺少绘图库时由调用方捕获并仅记录。
    """
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/robot_process_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    actions = [action for action in actions if action != 'done']
    if not actions:
        raise ValueError("当前面没有可视化动作")
    issue_ids = set(issue_ids or [])
    physical_risks = {
        action['id']: action.get('_physical_support')
        for action in actions
        if isinstance(action.get('_physical_support'), dict)
        and action['_physical_support'].get('risk')
    }
    physical_issue_ids = set(physical_risks)
    other_issue_ids = issue_ids - physical_issue_ids
    output_dir = output_dir or resolve_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename + '.png')

    box_types = list(dict.fromkeys(str(a.get('box_type', 'unknown'))
                                  for a in actions))
    fixed_colors = {'105': '#4C9BE8', '203': '#F59E42'}
    palette = ['#4C9BE8', '#F59E42', '#59A14F', '#E15759',
               '#B07AA1', '#76B7B2', '#EDC948', '#9C755F']
    type_colors = {
        box_type: fixed_colors.get(box_type, palette[index % len(palette)])
        for index, box_type in enumerate(box_types)
    }

    def cuboid_faces(x, y, z, dx, dy, dz):
        """返回轴对齐长方体的六个四边形面。"""
        points = [
            (x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
            (x, y, z + dz), (x + dx, y, z + dz),
            (x + dx, y + dy, z + dz), (x, y + dy, z + dz),
        ]
        return [[points[i] for i in face] for face in (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
        )]

    fig = plt.figure(figsize=(16, 9), dpi=120)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    ax3 = fig.add_subplot(grid[0, 1], projection='3d')
    max_x = max_y = max_z = 0.0

    for action in actions:
        action_id = action['id']
        box_type = str(action.get('box_type', 'unknown'))
        color = type_colors[box_type]
        boxes = _action_boxes(action)
        physical_risk = physical_risks.get(action_id)
        risk_box_indices = set(
            physical_risk.get('risk_box_indices', [])
            if physical_risk else [])
        for box_index, (x, y, z, length, width, height) in enumerate(
                boxes, start=1):
            ax.add_patch(Rectangle(
                (y, z), width, height,
                facecolor=color, edgecolor='white', linewidth=1.0, alpha=0.86))
            ax3.add_collection3d(Poly3DCollection(
                cuboid_faces(x, y, z, length, width, height),
                facecolors=color, edgecolors='white', linewidths=0.45, alpha=0.78))
            if box_index in risk_box_indices:
                # 正视图用粗红底边标记重心不稳定的单箱；三维图在其底面铺红色面。
                ax.plot(
                    [y, y + width], [z, z],
                    color='#D62728', linewidth=5.0,
                    solid_capstyle='butt', zorder=8)
                ax.add_patch(Rectangle(
                    (y, z), width, height, fill=False,
                    edgecolor='#D62728', linewidth=2.0, zorder=7))
                bottom_face = [[
                    (x, y, z + 1.0),
                    (x + length, y, z + 1.0),
                    (x + length, y + width, z + 1.0),
                    (x, y + width, z + 1.0),
                ]]
                ax3.add_collection3d(Poly3DCollection(
                    bottom_face, facecolors='#D62728',
                    edgecolors='#B71C1C', linewidths=1.4, alpha=0.62))
            max_x = max(max_x, x + length)
            max_y = max(max_y, y + width)
            max_z = max(max_z, z + height)

        y0 = min(box[1] for box in boxes)
        y1 = max(box[1] + box[4] for box in boxes)
        z0 = min(box[2] for box in boxes)
        z1 = max(box[2] + box[5] for box in boxes)
        if physical_risk:
            edge_color = '#D62728'
            line_width = 3.2
            line_style = '-'
        elif action_id in issue_ids:
            edge_color = '#C2185B'
            line_width = 3.0
            line_style = '--'
        else:
            edge_color = '#263238'
            line_width = 1.4
            line_style = '-'
        ax.add_patch(Rectangle(
            (y0, z0), y1 - y0, z1 - z0, fill=False,
            edgecolor=edge_color, linewidth=line_width,
            linestyle=line_style))
        support_text = ''
        if physical_risk:
            support_text = (
                f'\nsupport {physical_risk["support_ratio"] * 100:.0f}%'
                f' / min {physical_risk["min_box_support_ratio"] * 100:.0f}%')
        ax.text(
            (y0 + y1) / 2, (z0 + z1) / 2,
            f'#{action_id}\n{box_type} x{sum(action["num"])}{support_text}',
            ha='center', va='center', fontsize=8, fontweight='bold', color='#15202B')
        ax3.text(
            max(box[0] + box[3] for box in boxes) + 12,
            (y0 + y1) / 2, (z0 + z1) / 2,
            f'#{action_id}', fontsize=8, fontweight='bold', color=edge_color)

    car_width = float(car_size['W'])
    car_height = float(car_size['H'])
    ax.axvline(car_width, color='#555', linestyle=':', linewidth=1.3)
    ax.set_xlim(-40, max(car_width + 40, max_y + 80))
    ax.set_ylim(-30, min(car_height + 40, max_z + 120))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Y / truck width (mm)')
    ax.set_ylabel('Z / height (mm)')
    ax.set_title('Front view (Y-Z), labels are grab IDs', fontsize=13, pad=10)
    ax.grid(True, alpha=0.18)

    ax3.set_xlim(0, max(600, max_x + 100))
    ax3.set_ylim(0, max(car_width, max_y + 100))
    ax3.set_zlim(0, min(car_height, max_z + 120))
    ax3.set_box_aspect((max(600, max_x + 100),
                        max(car_width, max_y + 100),
                        min(car_height, max_z + 120)))
    ax3.set_xlabel('X depth')
    ax3.set_ylabel('Y width')
    ax3.set_zlabel('Z height')
    ax3.set_title('3D carton layout', fontsize=13, pad=10)
    ax3.view_init(elev=23, azim=-56)

    legends = [Patch(facecolor=type_colors[t], label=f'Type {t}') for t in box_types]
    if physical_issue_ids:
        legends.append(Patch(
            facecolor='none', edgecolor='#D62728', linewidth=2.5,
            linestyle='-', label='Physical support risk'))
    if other_issue_ids:
        legends.append(Patch(
            facecolor='none', edgecolor='#C2185B', linewidth=2.5,
            linestyle='--', label='Path / other risk'))
    fig.legend(handles=legends, loc='lower center', ncol=max(1, len(legends)),
               frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=10)
    fig.suptitle(title or filename, fontsize=16, fontweight='bold', y=0.97)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.06, right=0.98)
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


class Plot(object):
    """累积 Plotly 图元并把单抓路径场景写入 HTML。"""
    def __init__(self, filename, output_dir=None):
        """创建空图；``filename`` 不含扩展名，默认保存到当天日志目录。"""
        if py is None or go is None:
            raise RuntimeError("plotly 未安装，无法生成逐抓 HTML 可视化")
        output_dir = output_dir or resolve_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(output_dir, filename + ".html")
        self.data = []
        self.layout = {'title': 'Plot',
                       'showlegend': False
                       }

        self.fig = {'data': self.data,
                    'layout': self.layout}

    def plot_tree(self, X, trees):
        """按搜索空间维数选择二维或三维方式绘制树边。"""
        if X.dimensions == 2:  # plot in 2D
            self.plot_tree_2d(trees)
        elif X.dimensions == 3:  # plot in 3D
            self.plot_tree_3d(trees)
        else:  # can't plot in higher dimensions
            print("Cannot plot in > 3 dimensions")

    def plot_tree_2d(self, trees):
        """把各二维树的父子边追加为 Plotly 线段。"""
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
        """把各三维树的父子边追加为 Plotly 线段。"""
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
        """绘制轴对齐障碍物；二维用矩形，三维用半透明长方体。"""
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
        """按搜索空间维数绘制橙色起点。"""
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
        """按搜索空间维数绘制绿色目标点。"""
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
        """把已累积的图元写入 HTML；``auto_open`` 控制是否自动打开。"""
        py.offline.plot(self.fig, filename=self.filename, auto_open=auto_open)

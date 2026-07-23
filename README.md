# truck_loading_ws

卡车自动装载 ROS 2 工作区，包含激光雷达接入、点云采集与检测、车厢角点/宽度计算、垛序解析、机器人路径规划及机器人通信等功能。

当前开发环境以 Ubuntu 24.04、ROS 2 Jazzy、Python 3.12 为基准。现场电脑如果使用其他
Ubuntu/ROS 版本，应保证虚拟环境与该系统的 `/usr/bin/python3` 主版本一致，不能直接复制
另一台电脑创建好的 `venv`。

## 功能包

| 功能包 | 主要作用 | 主要入口 |
| --- | --- | --- |
| `hesai_ros_driver` | 禾赛激光雷达驱动，发布雷达点云 | `hesai_ros_driver_node` |
| `pointcloud_subscriber` | 订阅 `/lidar_points1`，合并若干帧并保存 PCD | `pointcloud_save_node` |
| `pointcloud_process` | 独立的点云角点及垛面检测工具 | `stacking_detection_node` |
| `cabin_width_check` | 多站点云拼接、车厢宽度切片测量、超限判断及 RViz 显示 | `width_check_node` |
| `robot_process` | 垛型接收、垛序解析、角点/垛面检测、路径规划、断点续传及机器人 TCP 通信 | `robot_process_node` |

## 目录说明

```text
truck_loading_ws/
├── src/                 ROS 2 功能包源码
├── test_data/           离线测试数据
├── build/               colcon 构建中间文件
├── install/             colcon 安装结果
└── log/                 构建日志、运行日志和断点续传数据
```

`build/`、`install/`、`log/`、PCD 点云及 Python 缓存均为本地生成内容，不应提交到 Git。

## 环境准备

### 新电脑首次部署

先确认系统 Python、ROS 版本和工作空间位置：

```bash
/usr/bin/python3 --version
source /opt/ros/jazzy/setup.bash
echo "$ROS_DISTRO"
pwd
```

安装系统依赖。OpenCV 使用系统包，便于与 ROS 及可视化组件共用：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libgl1 libglib2.0-0
```

在工作空间外创建虚拟环境。必须使用系统的 `/usr/bin/python3`，并启用
`--system-site-packages`，否则虚拟环境无法找到通过 APT 安装的 `cv2`、`rclpy` 和 ROS 消息包：

```bash
/usr/bin/python3 -m venv --system-site-packages ~/venv
source ~/venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
cd ~/workcells/truck_loading_ws
```

OpenCV 4.5/4.6 等系统包是按照 NumPy 1.x 接口编译的。项目环境将 NumPy 固定在 2.0 以下，
避免出现 `_ARRAY_API not found` 或 `numpy.core.multiarray failed to import`：

```bash
python -m pip install "numpy>=1.26.4,<2"
python -m pip install -r src/robot_process/requirements.txt
```

如果从仓库根目录统一安装 ROS 包依赖，也可以执行：

```bash
rosdep install --from-paths src --ignore-src -r -y
```

`robot_process` 的主要 Python 依赖记录在 `src/robot_process/requirements.txt`，包括 NumPy、
Open3D、Rtree、Plotly、grpcio、protobuf 和 colorlog。ROS 消息、`rclpy`、OpenCV 等依赖
由 ROS 2 或系统环境提供。

### 环境验证

安装完成后不要只检查 `pip list`，应实际导入节点用到的模块：

```bash
python -c "import sys; print('Python:', sys.version); print('Executable:', sys.executable)"
python -c "import numpy; print('NumPy:', numpy.__version__, numpy.__file__)"
python -c "import cv2; print('OpenCV:', cv2.__version__, cv2.__file__)"
python -c "import open3d; print('Open3D:', open3d.__version__)"
python -c "import rclpy; print('rclpy: OK')"
```

OpenCV 的版本属性是 `cv2.__version__`，`version` 两侧各有两个下划线。

推荐版本组合：

| 组件 | 推荐值 | 说明 |
| --- | --- | --- |
| Python | 与 `/usr/bin/python3` 一致 | 系统 OpenCV、ROS Python 扩展与具体 Python 主版本绑定 |
| NumPy | `1.26.4` | 当前 OpenCV/Open3D 组合稳定，禁止自动升级到 2.x |
| OpenCV | 系统 APT 版本 | Ubuntu 22.04 常见 4.5.4，Ubuntu 24.04 常见 4.6.0 |
| Open3D | `>=0.19.0` | 用于角点及垛面检测 |

### 已有虚拟环境找不到 `cv2`

先确认系统 Python 可以导入 OpenCV：

```bash
/usr/bin/python3 -c "import cv2; print(cv2.__version__)"
```

如果系统可以导入、虚拟环境不可以，检查：

```bash
source ~/venv/bin/activate
cat "$VIRTUAL_ENV/pyvenv.cfg"
```

若其中为 `include-system-site-packages = false`，改成 `true` 后重新进入环境：

```bash
sed -i \
  's/include-system-site-packages = false/include-system-site-packages = true/' \
  "$VIRTUAL_ENV/pyvenv.cfg"
deactivate
source ~/venv/bin/activate
```

如果虚拟环境的 Python 主版本与 `/usr/bin/python3` 不一致，不要继续复用该环境，应使用上面的
首次部署命令重新创建。系统 OpenCV 的 `.so` 文件不能跨 Python 3.10/3.12 直接加载。

### OpenCV 与 NumPy 2.x 冲突

出现以下信息时，说明系统 OpenCV 是按照 NumPy 1.x 编译的，但运行时加载了 NumPy 2.x：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

在虚拟环境中修复：

```bash
source ~/venv/bin/activate
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

如果连 `/usr/bin/python3` 也出现同样错误，通常是用户目录或 `/usr/local` 中安装的 NumPy 2.x
覆盖了 APT 版本。先查看实际加载位置：

```bash
/usr/bin/python3 -c "import numpy; print(numpy.__version__, numpy.__file__)"
```

若路径位于 `~/.local/` 或 `/usr/local/`，卸载这份 pip NumPy，再恢复系统包：

```bash
/usr/bin/python3 -m pip uninstall -y numpy
sudo apt install --reinstall python3-numpy python3-opencv
/usr/bin/python3 -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

系统 Python 中不要再用 pip 安装 NumPy 2.x；项目需要的 1.26.4 只安装在虚拟环境中。

### 每个终端的加载顺序

先加载 ROS 2 环境，再进入虚拟环境：

```bash
source /opt/ros/jazzy/setup.bash
source ~/venv/bin/activate
cd ~/workcells/truck_loading_ws
source install/setup.bash
```

## 构建工作区

在工作区根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

只构建机器人主流程：

```bash
colcon build --symlink-install --packages-select robot_process
source install/setup.bash
```

修改 Python 文件后，使用 `--symlink-install` 通常不需要重复复制源码；修改 proto、C++、安装资源或依赖配置后应重新构建。

## 常用启动命令

每个新终端都需要先执行：

```bash
source /opt/ros/jazzy/setup.bash
source ~/venv/bin/activate
source install/setup.bash
```

启动机器人主流程：

```bash
ros2 run robot_process robot_process_node
```

启动车厢宽度检测及 RViz：

```bash
ros2 launch cabin_width_check cabin_width.launch.py
```

离线回放车厢点云：

```bash
ros2 launch cabin_width_check cabin_width.launch.py \
  offline_pcd_dir:=/absolute/path/to/pcd_dir \
  use_rviz:=true
```

启动独立垛面检测节点：

```bash
ros2 run pointcloud_process stacking_detection_node
```

启动点云保存节点：

```bash
ros2 run pointcloud_subscriber pointcloud_save_node
```

启动禾赛雷达节点前，应先核对 `src/HesaiLidar_ROS_2.0/config/config.yaml` 中的雷达 IP、主机 IP、端口及点云话题配置：

```bash
ros2 run hesai_ros_driver hesai_ros_driver_node
```

## robot_process 运行方式

主配置文件位于 `src/robot_process/robot_process/config.json`。常用参数包括：

| 参数 | 作用 |
| --- | --- |
| `ip`、`port` | 机器人 TCP 服务监听地址和端口，当前默认端口为 `8001` |
| `off_line_mode` | `false` 时通过 gRPC 接收垛型，`true` 时读取本地 PKL |
| `use_corner` | 是否启用角点检测补偿 |
| `chk_enable` | 是否启用垛面检测流程 |
| `show_env` | 是否为每一抓生成路径规划 HTML 可视化 |
| `resume_save` | 是否保存断点进度 |
| `resume_on_restart` | 重启后是否自动检查并恢复进度 |
| `resume_need_confirm` | 断点恢复前是否等待人工确认 |

在线模式下，程序在 `5007` 端口接收规划器 gRPC 垛型数据，并在配置的 TCP 端口等待机器人连接。离线模式从 `src/robot_process/robot_process/pkl_data/` 读取垛型文件。

执行 `cmd_chk_path` 时，即使 `show_env=false`，混装 Block 也会为每一抓保存一份
HTML 路径可视化；每面完成后仍会保存整面 PNG。检查结束后，完整汇总会同时
写入 TXT、JSON，并打印到终端和运行日志中。

垛面检测点云默认保存到当前 `robot_process` 功能包所属工作空间的
`log/robot_process/pcd_logs/`。程序通过包自身路径定位工作空间，不使用其他 overlay
工作空间的优先顺序。需要固定到指定磁盘或目录时，在启动节点前设置：

```bash
export ROBOT_PROCESS_PCD_DIR=/absolute/path/to/pcd_logs
ros2 run robot_process robot_process_node
```

## 当前混装面接口

混装面使用扁平的 `Mixture` 列表，不再额外嵌套 `Items`：

```proto
message Mixture {
    string Type = 1;
    int32 Num = 2;
    Position Pos = 3;
}

message Position {
    float X = 1; // 车厢深度方向
    float Y = 2; // 车厢宽度方向
    float Z = 3; // 车厢高度方向
}
```

示例：

```json
"mixture": [
  {"Type": "104", "Num": 4, "Pos": {"X": 20, "Y": 0, "Z": 0}},
  {"Type": "201", "Num": 4, "Pos": {"X": 30, "Y": 1360, "Z": 580}}
]
```

同一个 `Block` 内的全部 `Mixture` 条目组成一个混装面，列表顺序就是放置动作顺序。每条数据会生成对应的 `rp.boxes` 来料记录和 `rp.robot_offsets` 放置动作，并携带当前箱型及其有效尺寸。

`Type` 必须能在订单箱型信息中找到；`Num` 必须大于零且不能超过对应箱型的 P1 单抓能力；坐标单位为毫米，并且必须位于车厢范围内。混装面的 `area_cfg` 全部固定为 `1`；`dir` 根据整抓箱组的中心 Y 坐标判断，中心位于车厢左半边时为 `1`，位于右半边时为 `2`。

正式接口文件位于 `src/robot_process/robot_process/grpc_pkg/interface/proto/`。修改 proto 后，需要重新生成 Python 接口文件并重新构建功能包。

## 机器人 TCP 关键接口

机器人响应中的每个数据块为 41 字节，浮点数据按照大端 `float32` 编码。

### `cmd_get_pallet`

当前返回 3 个数据块：

| 数据块 | 内容 |
| --- | --- |
| 1 | 当前 block 总箱数、码垛面数、车厢宽度 |
| 2 | 当前 block 默认箱型的有效尺寸 `L/W/H` |
| 3 | 混装 block 在完整 `rp_list` 中的位置 |

混装 block 位置从 `1` 开始编号：不存在混装 block 时为 `0`，`rp_list[0]` 是混装 block 时为 `1`，`rp_list[1]` 是混装 block 时为 `2`。如果存在多个混装 block，发送第一个的位置。

### `cmd_get_box`

当前返回 2 个数据块：

| 数据块 | 内容 |
| --- | --- |
| 1 | 来料配方号 `box_cfg`、箱型 `box_type`、位置编号 `area_cfg` |
| 2 | 当前这一抓箱子的有效尺寸 `L/W/H` |

有效尺寸为原始箱体尺寸加配置中的预留量。混装面可以在不同抓之间切换箱型，尺寸取自当前弹出的 `rp.boxes` 记录，不使用当前 block 的固定默认尺寸。

## 车厢宽度检测

在线模式的基本流程：

1. 机器人移动到一个采集站点。
2. 调用 `/cabin/capture_station` 保存当前双雷达点云。
3. 依次移动并重复采集，直到车尾。
4. 调用 `/cabin/analyze` 拼接点云、逐片测宽并发布结果。
5. 调用 `/cabin/reset` 清空本轮数据。

配置文件位于 `src/cabin_width_check/config.json`，RViz 配置位于 `src/cabin_width_check/rviz/cabin_width.rviz`。

## 测试

构建后运行功能包测试：

```bash
colcon test
colcon test-result --verbose
```

单独运行混装面测试：

```bash
PYTHONPATH=src/robot_process/robot_process \
python3 -m unittest -v src/robot_process/test/test_mixture.py
```

## 常见问题

- 找不到 ROS 包或节点：确认当前终端已经加载 `/opt/ros/jazzy/setup.bash` 和本工作区 `install/setup.bash`。
- 找不到 `open3d`、`rtree` 或 `grpc`：确认运行节点的 Python 与安装依赖时使用的是同一个环境。
- 虚拟环境找不到 `cv2`：确认 `pyvenv.cfg` 中为 `include-system-site-packages = true`，并确认虚拟环境与 `/usr/bin/python3` 主版本相同。
- OpenCV 报 `_ARRAY_API not found`：当前加载了 NumPy 2.x，按照“OpenCV 与 NumPy 2.x 冲突”一节降级到 NumPy 1.26.4。
- 在线模式一直等待：依次检查规划器是否连接 gRPC `5007`、机器人是否连接 TCP `8001`，以及防火墙和 IP 配置。
- 点云没有数据：检查雷达网络参数、实际发布话题以及节点订阅的 `/lidar_points1`、`/lidar_points2` 是否一致。
- 修改 proto 后字段未生效：重新生成 `*_pb2.py`，再重新构建并加载工作区。

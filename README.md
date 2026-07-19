# truck_loading_ws

卡车自动装载 ROS 2 工作区，包含激光雷达接入、点云采集与检测、车厢角点/宽度计算、垛序解析、机器人路径规划及机器人通信等功能。

当前开发环境以 ROS 2 Jazzy、Python 3.12 为基准。

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

先加载 ROS 2 环境：

```bash
source /opt/ros/jazzy/setup.bash
```

`robot_process` 的主要 Python 依赖记录在 `src/robot_process/requirements.txt`：

```bash
python3 -m pip install -r src/robot_process/requirements.txt
```

主要依赖包括 NumPy、Open3D、Rtree、Plotly、grpcio、protobuf 和 colorlog。ROS 消息、`rclpy`、PCL 等依赖由 ROS 2 或系统环境提供。

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
| `show_env` | 是否显示路径规划环境 |
| `resume_save` | 是否保存断点进度 |
| `resume_on_restart` | 重启后是否自动检查并恢复进度 |
| `resume_need_confirm` | 断点恢复前是否等待人工确认 |

在线模式下，程序在 `5007` 端口接收规划器 gRPC 垛型数据，并在配置的 TCP 端口等待机器人连接。离线模式从 `src/robot_process/robot_process/pkl_data/` 读取垛型文件。

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
- 在线模式一直等待：依次检查规划器是否连接 gRPC `5007`、机器人是否连接 TCP `8001`，以及防火墙和 IP 配置。
- 点云没有数据：检查雷达网络参数、实际发布话题以及节点订阅的 `/lidar_points1`、`/lidar_points2` 是否一致。
- 修改 proto 后字段未生效：重新生成 `*_pb2.py`，再重新构建并加载工作区。

# truck_loading_ws

厢式货车自动装载应用，使用ROS2工作区，包含激光雷达接入、点云采集与检测、车厢角点/宽度计算、垛序解析、机器人路径规划及机器人通信等功能。

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
| `chk_enable` | 是否启用路径干涉、左右余量及车厢顶部检查 |
| `show_env` | 是否为每一抓生成路径规划 HTML 可视化 |
| `mixture_support_z_tolerance` | 混装支撑面允许的上下高度误差，单位 mm，默认 20 |
| `mixture_support_min_ratio` | 混装整抓支撑率诊断参考值，默认 0.8；当前不单独触发告警 |
| `mixture_support_min_box_ratio` | 混装单箱支撑率诊断参考值，默认 0.6；当前不单独触发告警 |
| `reserve_grip` | 手爪碰撞包围盒余量；当前路径高度使用第3项，以 `config.json` 当前值为准 |
| `resume_save` | 是否保存断点进度 |
| `resume_on_restart` | 重启后是否自动检查并恢复进度 |
| `resume_need_confirm` | 断点恢复前是否等待人工确认 |

在线模式下，程序在 `5007` 端口接收规划器 gRPC 垛型数据，并在配置的 TCP 端口等待机器人连接。离线模式从 `src/robot_process/robot_process/pkl_data/` 读取垛型文件。

执行 `cmd_chk_path` 时，即使 `show_env=false`，混装 Block 也会为每一抓保存一份
HTML 路径可视化；每面完成后仍会保存整面 PNG。检查结束后，完整汇总会同时
写入 TXT、JSON，并打印到终端和运行日志中。订单中收到的原生 `mixture`
字段也会写入检查汇总，不额外展开混装逐抓规划细节。

`cmd_chk_path` 使用独立的垛序和环境副本，不消费正式 `cmd_get_box`、
`cmd_get_path` 队列，也不写断点游标。返回值仍为 `float`，数值改为三位状态码，
三位依次表示“路径 / 垛序 / PLC字段”：`1` 为通过，`2` 为失败。例如 `111`
表示全部通过，`112` 表示路径和垛序通过、PLC字段失败。垛序检查会逐抓核对
get_box 与 get_path 的箱型、数量编码、尺寸、区域、面号和动作标志；PLC检查会
按整车来箱序号复算 `BoxP3Ind`（不翻转序号）和 `BoxRightInd`（需要右翻的
1XX-P1最右抓；混装面以及统一靠左、不右翻的尾料/尾门P1简单行不参与）。
尾门区域如果存在常规箱P3，仍按正常规则加入 `BoxP3Ind`。

启用 `chk_enable` 后，混装面还会按实际放置顺序检查每一抓的底面支撑情况。
该检查只告警和记录，不会阻止路径下发。当前风险判据是单箱重心是否落在直接
接触面内，或至少两个支撑面的联合凸包内；支撑面积比例及配置阈值只作为诊断
信息保存。整抓支撑率、最低单箱支撑率和风险单箱编号会写入检查汇总；面级 PNG
使用红色实线框出风险抓，并以粗红底边标记重心不稳定的单箱。

订单只有在全部 Block 成功构造垛序后才会归档；仅通过 gRPC 接收到订单时不会
保存。每次成功解析生成一个独立 JSON，目录为
`log/robot_process/YYYYMMDD/orders/`。在线订单的归档文件会同时保存规划器
原始请求中的完整 `answer`、`condition`，以及程序归一化后的 `order_fields`；
离线文件或旧断点恢复无法取得原始请求时，`answer`、`condition` 记录为 `null`。

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
    bool Ishead = 4; // 当前混装面是否位于异形车头
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

`Block` 本身没有异形车头字段，标记位于下一层的 `Regular.Ishead`、
`Mixture.Ishead` 或 `Trapezoid.Ishead`。车辆条件中的 `CarCondition.head`
保存车头前端的 `L/W/H` 尺寸。程序同时使用条目标记和有效的 `head`
尺寸识别异形区域：面序按从车头到车尾处理，下一面的起点纵深由上一面
实际占用纵深推进，不按面数平均。常规/梯形面使用箱子物理长度；混装面
使用所有条目 `Pos.X + 箱长` 的最大值。当前面使用其靠车头一侧的最窄
车宽，超过 `head.L` 后恢复并限制为 `original.W`。面级宽度会用于垛序、
APP、侧壁边界、路径校验和垛面检测参数。

异形标记、累计纵深和每面车宽会进入规范化订单归档；前6个处于异形区域
的 block 序号通过垛型总体信息第4数据块发送给机器人。若存在 `Ishead=true`
却缺少有效的 `head.L/head.W`，订单会因无法计算车头斜率而拒绝构造垛序。

`Type` 必须能在订单箱型信息中找到；`Num` 必须大于零且不能超过对应箱型的单抓能力；坐标单位为毫米，并且必须位于车厢范围内。混装 P1 抓的 `area_cfg` 通常为 `1`；若当前抓是当前高度最靠近车壁的一抓，并且内侧已码箱体的顶面高出本次放置底面超过 50 mm，则记为当前高度收尾位置 `4`，不再使用 `5`。混装动作先按整抓中心所在的车宽半区生成初始 `dir`；执行 `cmd_get_path` 时，P1 抓会根据当前抓 X/Z 范围内已码箱体的左右实际间隙重新选择方向，分析失败时才回退初始值。

正式接口文件位于 `src/robot_process/robot_process/grpc_pkg/interface/proto/`。修改 proto 后，需要重新生成 Python 接口文件并重新构建功能包。

## 机器人 TCP 关键接口

机器人请求命令及用途如下：

| 命令 | 十六进制请求 | 作用 |
| --- | --- | --- |
| `cmd_get_pallet` | `fefe0000010100000000000000000000` | 读取当前 Block 的总体垛型信息 |
| `cmd_get_per_count` | `fefe0000010200000000000000000000` | 读取当前码垛面的抓数、箱型和角点收缩量 |
| `cmd_get_box` | `fefe0000010300000000000000000000` | 读取下一抓的来料配方和箱子尺寸 |
| `cmd_get_path` | `fefe0000010400000000000000000000` | 读取下一抓的完整放置路径 |
| `cmd_chk_path` | `fefe0000010500000000000000000000` | 独立批量检查整单路径、垛序和PLC字段 |
| `cmd_stacking` | `fefe0000010600000000000000000000` | 触发双雷达垛面检测并读取宽度结果 |

每个响应数据块固定为41字节：第1字节是数据块标记，随后4字节是
`class_id`，最后是9个大端`float32`槽位，共36字节。本文用`float[0]`
到`float[8]`表示这9个槽位。普通信息块的数据块标记和`class_id`均为0，
未使用的float槽位也全部补0；路径点块的定义单独见`cmd_get_path`。

### `cmd_get_pallet`

固定返回4个数据块。

第1块：当前Block总体信息

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]` | `block_box_count` | 当前Block包含的箱子总数，不是抓数 | 箱 |
| `float[1]` | `face_count` | 当前Block包含的码垛面数量 | 面 |
| `float[2]` | `car_width` | 订单中的车厢宽度`CarCondition.original.W` | mm |
| `float[3]` | `head_width` | 异形车头最前端宽度`CarCondition.head.W` | mm；未配置为0 |
| `float[4]` | `head_length` | 异形车头区域长度`CarCondition.head.L` | mm；未配置为0 |
| `float[5]` | `frame_width` | 尾门门框宽度`CarCondition.frame.W` | mm；未配置为0 |
| `float[6]`～`float[8]` | 保留 | 当前未使用 | 0 |

第2块：当前Block默认箱型的有效尺寸

| 槽位 | 字段 | 定义 | 单位 |
| --- | --- | --- | --- |
| `float[0]` | `box_length` | 箱子沿车厢深度方向的有效长度L | mm |
| `float[1]` | `box_width` | 箱子P1姿态沿车厢宽度方向的有效宽度W | mm |
| `float[2]` | `box_height` | 箱子P1姿态的有效高度H | mm |
| `float[3]`～`float[8]` | 保留 | 当前未使用 | 0 |

“有效尺寸”是原始箱体尺寸加订单中的箱体预留量。混装Block中该块仅表示
Block默认箱型；每一抓的实际箱型尺寸应以`cmd_get_box`第2块为准。

第3块：整单中的混装Block位置

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]`～`float[5]` | `mixture_block_index_1`～`6` | 前6个混装Block在完整Block列表中的序号 | 从1开始；不足部分为0 |
| `float[6]`～`float[8]` | 保留 | 当前未使用 | 0 |

第4块：整单中的异形车头Block位置

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]`～`float[5]` | `head_block_index_1`～`6` | 前6个异形车头Block在完整Block列表中的序号 | 从1开始；不足部分为0 |
| `float[6]`～`float[8]` | 保留 | 当前未使用 | 0 |

混装和异形车头位置都按完整Block列表从1开始编号：第一个Block对应1，
第二个Block对应2。不存在对应Block时6个位置全部为0；超过6个时只发送前6个。

除下一层垛型条目的显式
`Ishead=true` 外，`CarCondition.head` 有效时也会根据累计纵深自动识别；
两种方式均未识别到异形区域时，第4块全部为0。

### `cmd_get_per_count`

固定返回1个数据块，并且不会消费来料或路径队列。

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]` | `face_grab_count` | 当前面全部P1、P2、P3动作的总放置抓数 | 抓；无待处理来料时为1 |
| `float[1]` | `box_type` | 当前待取抓的箱型代号 | 数字形式，如105、203 |
| `float[2]` | `p1_bottom_row_grab_count` | 当前面P1最低层的单行抓数，不是箱数 | 抓；无P1或队列为空时为1，解析异常时为0 |
| `float[3]` | `corner_shrink_mm` | 异形车头左右角点各自需要向内收缩的单侧距离 | mm；普通区域为0 |
| `float[4]`～`float[8]` | 保留 | 当前未使用 | 0 |

角点单侧收缩量按`(original.W - 当前面可用车宽) / 2`计算；普通区域、
超过异形车头区域或无当前面时为`0.0`。当前面优先取`boxes[0]`；来料队列
为空时，仅角点收缩的面信息回退到下一条未执行路径，总抓数和P1最低层
单行抓数按兼容规则发送`1`。重复请求不会消费来料或路径队列，也不会累计
角点收缩量。

### `cmd_get_box`

固定返回2个数据块。每次成功返回会消费当前Block来料队列中的下一抓。

第1块：来料配方

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]` | `box_cfg` | 当前抓箱数及抓取方式编码 | 见下方数量编码规则 |
| `float[1]` | `box_type` | 当前抓的箱型代号 | 数字形式，如105、203 |
| `float[2]` | `area_cfg` | 当前抓在本行中的位置或特殊收尾标记 | 见下方位置编码规则 |
| `float[3]`～`float[8]` | 保留 | 当前未使用 | 0 |

`box_cfg`数量编码：

| 条件 | 发送值 |
| --- | --- |
| 非混装面、1XX箱型、P1区域、同面同高度恰好三抓，并且是物理最右抓 | 实际箱数+20；常规垛和梯形垛均适用 |
| 2XX或3XX箱型的任意区域 | 实际箱数+10 |
| 1XX箱型的P3区域 | 实际箱数+10 |
| 其他情况 | 实际箱数 |

“实际箱数+20”的优先级高于“实际箱数+10”。物理最右抓按放置Y坐标
判断，不按执行顺序判断。尾门梯形仍按左、中、右顺序执行，只是最右抓使用
+20编码；混装面、P2、P3以及2XX/3XX不会使用+20编码。

`area_cfg`位置编码：

| 数值 | 定义 |
| --- | --- |
| `1` | 当前行物理最左抓、单抓行、尾料、尾门简单行或P3；异形车头P1三抓的物理最右抓也固定为1 |
| `2` | 三抓及以上行中，不在最左或最右位置的中间抓 |
| `3` | 普通常规/梯形行的物理最右抓 |
| `4` | 混装面当前高度靠近车壁的收尾抓，并且内侧已有箱体顶面高出本次放置底面超过50mm |
| `11`或`13` | Group/Stack两抓行的执行末抓；在原物理位置码1或3上加10 |

混装面除满足`area_cfg=4`的收尾抓外均发送1，不使用2、3、5。`area_cfg`
与`box_cfg`相互独立，例如尾门梯形P1三抓的物理最右抓可以同时是
`box_cfg=实际箱数+20`、`area_cfg=1`。

第2块：当前抓箱子的有效尺寸

| 槽位 | 字段 | 定义 | 单位 |
| --- | --- | --- | --- |
| `float[0]` | `box_length` | 当前抓箱型的有效长度L | mm |
| `float[1]` | `box_width` | 当前抓箱型的有效宽度W | mm |
| `float[2]` | `box_height` | 当前抓箱型的有效高度H | mm |
| `float[3]`～`float[8]` | 保留 | 当前未使用 | 0 |

有效尺寸为原始箱体尺寸加配置中的预留量。混装面可以在不同抓之间切换箱型，尺寸取自当前弹出的 `rp.boxes` 记录，不使用当前 block 的固定默认尺寸。

`box['action']`不通过该命令发送，结束动作只在`cmd_get_path`最后的信息块中发送。

### `cmd_get_path`

返回块数随当前抓路径点数量变化：`路径点数量+1`。普通单段放置通常包含
`x0、x1、APP、goal_1`四个路径点，因此共返回5块；一抓分多段放置时，
每增加一个落点就再增加一个路径点块。每次成功返回会消费下一条路径动作。

路径点块顺序：

| 块序 | 名称 | 定义 |
| --- | --- | --- |
| 1 | `x0` | 初始悬停/过渡点 |
| 2 | `x1` | 第二过渡点，包含侧向避障及高度调整 |
| 3 | `APP` | 接近放置位置的进入点 |
| 4 | `goal_1` | 第一段箱子的最终落点 |
| 5及以后 | `goal_2...goal_n` | 一抓多段放置时的后续落点 |
| 最后1块 | `path_info` | 区域、结束动作和第一段箱数，不是坐标点 |

每个路径点块的定义相同：

| 块内项目 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| 第1字节 | `point_flag` | 路径点块标记 | 固定为1 |
| 4字节`class_id` | `path_check_status` | 只在第一个路径点块中携带路径检查结果 | `0`=未启用检查，`1`=通过，`2`=失败；其他路径点固定为0 |
| `float[0]` | `X` | 程序最终生成的X坐标 | mm |
| `float[1]` | `Y` | 程序最终生成的Y坐标 | mm |
| `float[2]` | `Z` | 程序最终生成的Z坐标 | mm |
| `float[3]`～`float[8]` | 保留 | 当前未使用 | 0 |

`path_check_status`只在配置`chk_enable=true`时计算。它反映当前这一抓的
路径检查结果，不等同于`cmd_chk_path`的整单三位状态。即使检查失败，当前
实现仍会返回路径，由机器人侧根据状态决定如何处理。

最后的`path_info`块使用普通信息块格式：

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]` | `area` | 当前抓的放置区域 | `1`=P1主堆区，`2`=P2侧壁区，`3`=P3侧立区 |
| `float[1]` | `action` | 当前抓完成后的流程动作 | `0`=继续当前面，`1`=当前面结束，`2`=当前Block结束，`3`=整单结束 |
| `float[2]` | `first_segment_box_count` | 当前抓第一段包含的箱数 | 箱；多段放置时不是整抓总箱数 |
| `float[3]`～`float[8]` | 保留 | 当前未使用 | 0 |

一抓多段放置由`num=[n1,n2,...]`和`gaps=[g1,...]`生成多个goal。当前报文
会发送全部落点坐标，但信息块只发送第一段箱数`n1`；`dir`、APP居中偏移、
角点补偿值和碰撞包围盒不会作为独立字段发送，它们已经反映在最终坐标中。

### `cmd_chk_path`

该命令不会逐抓向机器人返回路径。程序使用独立的垛序和环境副本检查完整
订单，整轮结束后固定返回1个普通信息块，不消费正式`cmd_get_box`和
`cmd_get_path`队列。

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]` | `check_status` | 路径、垛序、PLC字段三类检查的汇总状态 | 三位数字XYZ |
| `float[1]`～`float[8]` | 保留 | 当前未使用 | 0 |

三位状态的每一位只取1或2：

| 位 | 检查内容 | `1` | `2` |
| --- | --- | --- | --- |
| X（百位） | 路径检查 | 全部抓路径通过 | 至少一抓路径异常、系统异常或检查未完整执行 |
| Y（十位） | 垛序检查 | 来料信息与路径动作逐抓一致 | 数量编码、箱型、尺寸、区域、面号、动作等至少一项不一致 |
| Z（个位） | PLC字段检查 | `BoxP3Ind`、`BoxRightInd`均与程序推导一致 | 至少一个PLC序列不一致或字段缺失 |

例如`111`表示三类检查全部通过，`112`表示仅PLC字段失败，`211`表示仅
路径失败。检查初始化、汇总等流程发生无法分类的严重异常时，兜底返回`222`。
完整逐抓结果、异常定位和原生混装字段保存在本次检查目录的JSON/TXT中，
不会放入TCP响应数据块。

### `cmd_stacking`

固定返回1个普通信息块：

| 槽位 | 字段 | 定义 | 单位/编码 |
| --- | --- | --- | --- |
| `float[0]` | `status` | 本次双雷达采集和宽度计算结果 | `1`=成功，`2`=采集、计算或程序异常 |
| `float[1]` | `expected_width` | 当前抓的理论宽度 | mm |
| `float[2]` | `measured_width` | 点云计算得到的实际缺口/垛面宽度 | mm；失败时为0 |
| `float[3]`～`float[8]` | 保留 | 当前未使用 | 0 |

理论宽度通常按“当前抓实际箱数×单箱宽度”计算：P1使用箱子有效宽度W，
P3侧立使用箱子有效高度H。混装面会用真实缺口位置和宽度辅助点云候选筛选，
但机器人收到的`expected_width`仍保持上述理论抓宽。尚未成功执行过
`cmd_get_path`时理论宽度为`-1`；程序进入最外层异常兜底时发送
`status=2、expected_width=0、measured_width=0`。雷达采集超时、宽度计算
失败或检测到不能给出有效宽度的异常垛面时，发送`status=2`，测量值为0。

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

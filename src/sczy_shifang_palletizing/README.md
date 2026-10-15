# 四川中烟·什邡工厂装车功能包

ROS2包名：**sczy_shifang_palletizing**。由原 palletizing_sczy 独立迁移，
不修改工作空间内的 robot_process 或其他功能包。

当前组合：**什邡原版垛序 + 四点规则路径 + 原版通信协议**，不运行RRT/RRT*。

## 已恢复的原版垛序

- 第一层按左→中→右，下一层按右→中→左；奇偶按常规面号/层号及原梯形规则判断。
- 恢复原版每层统一dir，不采用新版“左→右→中”的顺序。
- 使用单箱型及 regular[0]，随后接原梯形列表；最大抓数固定为P1/P2/P3=4/2/2。
- 箱体使用original尺寸，不叠加box.reserve，不读取grip改变分抓。
- P3起点Y恢复50mm；梯形尾料恢复P3；尾门恢复从Y=0顺序放置，不做新版整面居中。
- 原版不生成组内分段放置，num只是转换为单元素列表供路径算法使用。
  Stack组内缝隙不因此变成一抓多段；此行为与原代码相同。
- 原版活跃的read_robot_offset不完整支持P2和E=3等情况。
  保留迁移已有的数量一致性检查：若声明箱数与实际动作不符，明确拒收，不静默丢箱。

原规则快照为 sczy_shifang_palletizing/rule_planning/legacy_sequence.py；
legacy_order.py只负责字段检查和动作结构适配。测试逐抓对照原版的数量、顺序、
dir、XYZ、面号及结束标志。scripts/rrt_env内的旧源文件仅用于回归参考，不参与运行。

## 保留的规则路径

固定路径为 x0 → x1 → APP → goal，不调用随机搜索，不生成RRT树。

- APP的Y偏移=(右侧间隙-左侧间隙)/2，限制在±100mm。
- 初始点、高度阈值、第一层处理及落点角点补偿保留上一轮迁移的规则。
- 角点补偿只用于dir=2、所在行最右抓且处于右半侧的落点，范围0～40mm。
- 路径发送成功后，真实/外扩箱体模型使用补偿后的落点；后续同行APP的左右
  间隙也采用已下发位置。原始垛序不回写，避免重复补偿；发送或检查失败不更新，
  换面/整单结束时清空本面记录。仍沿用旧协议的下发建模时序，不代表机器人到位确认。
- 原版dir是整行方向，方向一致性检查核对原垛序，不再用新版的贴边方向否决它。
- chk_enable=true时检查高度及过渡点/APP的箱体八角点、同层间隙。
  这不是完整连续扫掠检查，也不是机器人关节可达性检查。
- 旧报文没有路径失败状态位：失败时记录面/抓次、不下发该路径，并结束该连接。
  不自动重放旧订单，重新运行前须核对机器人已执行的动作。

配置路径：config/planner.json。此前迁移的初始点、安全余量保持不变：
P1 [650,1200,2000]，P2 [200,1600,1675]，P3 [650,1200,2400]，
手爪高度余量100mm、障碍物外扩49mm。上机前必须按什邡现场坐标系复核。

## 原通信协议保持不变

.proto文件及字段编号完全不变。两个生成的Python文件仅调整包导入路径，
序列化描述符及服务名称不变，不需要规划器/机器人升级协议。

每块仍为41字节：index(1)+class_id(4)+9个大端float(36)。
命令字、头部、保留位及0x91常量不变。

| 命令后缀 | 数据块 | 返回内容 |
| --- | --- | --- |
| 0101 | 2块 | 整单箱数/面数/车宽；原始箱长/宽/高 |
| 0102 | 1块 | 当前面总抓数、单字符箱型ord编码 |
| 0103 | 1块 | P1/P2实际箱数或P3配方号（1箱5、2箱6）、箱型编码 |
| 0104 | 4块 | x0/x1/APP/goal的XYZ；前三点class_id=P1/P2/P3区域码，末点=继续1/换面2/完成3 |

不增加0105/0106，不发送area_cfg、+10/+20数量编码或新版额外信息块。
原版没有混装面坐标和异形车头字段，不迁移这些场景。

## 新电脑首次创建工作空间

以下步骤在**新电脑**执行，以 `~/truck_loading_ws` 为工作空间，使用Bash终端。
这里只部署什邡包，不需要复制整个旧工作空间，也不需要安装robot_process。

### 1. 确认操作系统和ROS2

建议使用本包已验证的 Ubuntu 24.04 + ROS2 Jazzy + 系统Python 3.12。
若现场使用 Ubuntu 22.04 + ROS2 Humble + Python 3.10，以下命令中的
`jazzy` 全部替换为 `humble`；本包尚未在Humble实机验证。

若没有ROS2，先按对应官方说明完成ROS2安装（ROS-Base或Desktop均可）：
[Jazzy安装说明](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)、
[Humble安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)。
不要在同一终端混用两个ROS发行版或source其他旧工作空间。

~~~bash
source /opt/ros/jazzy/setup.bash
echo "$ROS_DISTRO"
/usr/bin/python3 --version
sudo apt update
sudo apt install python3-venv python3-pip python3-colcon-common-extensions python3-pytest
~~~

### 2. 创建ws并复制源码

~~~bash
mkdir -p ~/truck_loading_ws/src
cd ~/truck_loading_ws
~~~

将旧电脑的 **`src/sczy_shifang_palletizing` 整个目录**复制到新电脑的
`~/truck_loading_ws/src/` 下。可以通过U盘、文件管理器或scp传输。
不要复制旧电脑的build、install、log、venv；这些内容需要在新电脑重新生成。
最终应存在以下文件，而不是多嵌套一层同名目录：

~~~text
~/truck_loading_ws/src/sczy_shifang_palletizing/package.xml
~/truck_loading_ws/src/sczy_shifang_palletizing/setup.py
~/truck_loading_ws/src/sczy_shifang_palletizing/setup.cfg
~/truck_loading_ws/src/sczy_shifang_palletizing/requirements.txt
~/truck_loading_ws/src/sczy_shifang_palletizing/config/planner.json
~/truck_loading_ws/src/sczy_shifang_palletizing/launch/sczy_shifang_palletizing.launch.py
~/truck_loading_ws/src/sczy_shifang_palletizing/sczy_shifang_palletizing/node.py
~~~

本目录已经是功能包，**不用再运行 `ros2 pkg create`**。

### 3. 创建独立Python环境并安装依赖

使用系统Python创建环境，并允许访问系统/ROS依赖；不要直接复制旧电脑的虚拟环境，
也不要用Conda解释器替代ROS对应的系统解释器。解释器匹配要求见
[ROS官方Python环境说明](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Using-Python-Packages.rst)。
虚拟环境放在ws之外，避免colcon扫描到环境中的第三方包。

~~~bash
/usr/bin/python3 -m venv --system-site-packages ~/venvs/sczy_shifang
source ~/venvs/sczy_shifang/bin/activate
cd ~/truck_loading_ws
python -m pip install setuptools wheel -r src/sczy_shifang_palletizing/requirements.txt
python -c "import sys, rclpy, numpy, grpc, google.protobuf, plotly; print('Python:', sys.executable); print('NumPy:', numpy.__version__); print('依赖导入成功')"
~~~

只向此虚拟环境安装依赖，不使用 `sudo pip` 或 `--break-system-packages`。
本包不需要OpenCV、Open3D、Rtree；rclpy等ROS依赖由ROS安装提供。

### 4. 核对配置并构建

先核对 `src/sczy_shifang_palletizing/config/planner.json` 中的现场初始点、APP偏移、
手爪高度余量以及 `use_corner` 配置。不能把其他现场的坐标直接用于机器人。

~~~bash
cd ~/truck_loading_ws
python -m colcon list --base-paths src
python -m colcon build --base-paths src --packages-select sczy_shifang_palletizing
source install/setup.bash
ros2 pkg executables sczy_shifang_palletizing
~~~

最后一条应列出 `sczy_shifang_palletizing_node`。
这里明确用 `python -m colcon`，保证构建使用当前虚拟环境的解释器，避免直接调用
系统colcon时生成的节点入口绕过虚拟环境、运行时找不到依赖。

### 5. 离线检查与无机器人启动验证

~~~bash
python -B -m unittest discover -s src/sczy_shifang_palletizing/test -v
ros2 launch sczy_shifang_palletizing sczy_shifang_palletizing.launch.py enable_network:=false
~~~

`enable_network=false` 时不监听机器人/订单端口、不调用角点服务，不会下发机器人路径。
确认配置加载成功后按Ctrl+C退出。独立部署时，没有robot_process源码的对照测试允许跳过；
其他测试不应失败。此步骤仅验证软件部署，不代表实机运动安全验证通过。

### 6. 正式启动及之后每次打开新终端

每次都加载ROS、同一虚拟环境和当前工作空间：

~~~bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/sczy_shifang/bin/activate
cd ~/truck_loading_ws
source install/setup.bash
ros2 launch sczy_shifang_palletizing sczy_shifang_palletizing.launch.py
~~~

正式启动前确认以下事项：

- 规划器连接新电脑的gRPC端口5007；先成功下发订单，再让机器人连接TCP端口8001。
- 配置中的 `ip=0.0.0.0` 是本机监听地址，不是机器人的IP；机器人应连接新电脑的实际IP。
- 网络地址、防火墙及端口占用正确，不要让旧程序同时监听相同端口。
- `use_corner=true` 时，需要原角点服务可达，默认 `192.168.1.100:9002`。
  可在planner.json添加 `corner_ip`、`corner_port` 修改地址；本包不包含角点采集服务。
- 现场初始点和安全余量已经复核；当前路径检查不是完整连续碰撞或关节可达性检查。

默认日志在 `~/.ros/log/sczy_shifang_palletizing/`。
修改源码或默认配置后，重新执行第4步构建并重启节点。若需要直接读取源码目录中的配置，
可使用完整命令：

~~~bash
ros2 launch sczy_shifang_palletizing sczy_shifang_palletizing.launch.py \
  config_file:="$HOME/truck_loading_ws/src/sczy_shifang_palletizing/config/planner.json"
~~~

采用此参数时，配置修改只需重启节点，
代码修改仍需重新构建。工作空间或虚拟环境路径不要在构建后随意移动。

## 已有工作空间的构建与启动

支持ament_python形式；在本机ROS2 Jazzy/Python 3.12上验证，代码结构也适用于
Humble/Python 3.10，但未在Humble实机构建。请使用与ROS发行版匹配的Python解释器。

从工作空间根目录执行：

~~~bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/sczy_shifang/bin/activate
python -m colcon build --packages-select sczy_shifang_palletizing
source install/setup.bash
ros2 launch sczy_shifang_palletizing sczy_shifang_palletizing.launch.py
~~~

或者直接运行：

~~~bash
ros2 run sczy_shifang_palletizing sczy_shifang_palletizing_node
~~~

只验证ROS2节点、配置和安装，不监听订单/机器人端口：

~~~bash
ros2 launch sczy_shifang_palletizing sczy_shifang_palletizing.launch.py enable_network:=false
~~~

参数：

| 参数 | 默认 | 用途 |
| --- | --- | --- |
| config_file | 安装share目录的config/planner.json | 规划配置的绝对路径 |
| enable_network | true | false时不启动TCP/gRPC，但ROS节点保持运行 |
| robot_ip | 配置文件ip（0.0.0.0） | 机器人TCP监听地址 |
| robot_port | 配置文件port（8001） | 机器人TCP监听端口 |
| grpc_port | 5007 | 规划器接单端口 |
| log_dir | ~/.ros/log/sczy_shifang_palletizing | 运行日志和visualizations子目录 |
| offline_order_file | 空 | 本机可信pkl绝对路径；设置后启用离线订单 |

网络工作在独立线程，ROS执行器不被accept/recv阻塞；退出时主动关闭连接。
离线模式不再使用input()，一个离线订单结束后不自动整单重放。
包内7份历史pkl缺少E/Group/Stack等字段，原程序也无法解析，未擅自推测其垛型。

**默认8001/5007与robot_process可能冲突，不要同时按默认端口启动两包。**
需并行时，明确给本包设置其他端口，不修改其他包：

~~~bash
ros2 run sczy_shifang_palletizing sczy_shifang_palletizing_node --ros-args \
  -p robot_port:=8101 -p grpc_port:=5107
~~~

## 依赖检查

requirements.txt与setup.py一致，普通Python依赖为：

- numpy>=1.24,<2（Python 3.12应使用1.26.x，如1.26.4）。
- protobuf>=4.24.2,<7。
- grpcio>=1.56.2,<2。
- plotly>=5.15,<7（HTML显示）。

Rtree和colorlog已从运行依赖移除。旧搜索辅助代码、备份与旧日志保留在scripts/
和backup/，不随ROS2包安装。rclpy、launch、launch_ros、ament_index_python由ROS
发行版提供，不要通过普通pip替代ROS安装。

若使用venv，应匹配ROS的Python版本并能访问ROS模块，构建和启动使用同一环境；
仅在确实需要安装Python依赖时执行：

~~~bash
python -m pip install -r src/sczy_shifang_palletizing/requirements.txt
~~~

本次验证使用已有环境，没有修改系统Python或其他功能包的依赖。

## 回归测试

~~~bash
python -B -m unittest discover -s src/sczy_shifang_palletizing/test -v
~~~

离线测试包含原proto校验、原版垛序逐抓对照、规则路径高度边界/APP居中、
旧报文解析、整单命令模拟和失败队列保护。ROS部署测试另验证包发现、安装文件、
节点启动/退出与网络关闭。测试不向真实机器人发送路径。

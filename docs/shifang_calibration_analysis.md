# 什邡 AGV 导航归档：雷达与机械臂标定核查

核查日期：2026-10-16。对象为用户提供的 `/home/qinwentao/下载/ext_dense_feature.zip`。
归档 SHA256：`cf8ea0df6a3b47db8184800ce5d8ae17832ca2139ff73fa6d7becdcd6be34fe0`。
下文文件路径均相对于归档根目录；本次解压供阅读的位置为 `/tmp/shifang_calibration_review/`。

结论：按归档内 `pipeline_control/launch/run.launch` 启动，实际加载 `pipeline_control/config/basic_2.yaml`。
垛角点转换到 FANUC 机械臂基座使用其中的 `scanner_z_to_robot_base`，平移为
`(0.629, -0.469193, -0.558983) m`，四元数 `(qx,qy,qz,qw)` 为
`(0.00271, 0.0007761, 0.0072169, 0.99997)`。
当前编译的 `ext_dense_feature/src/tracker/tracker.cpp` 还硬编码了完全相同的七个数，用于日志中的坐标换算。

这是归档代码所选择的有效配置，包含现场调试后的修正；未取得现场运行中的参数快照、安装目录或二进制，不能据此证明现场今天仍使用相同版本。
本次仅阅读代码并导出核查资料，没有运行归档中的启动/安装/控制脚本，没有修改工作空间的运行配置或协议。

## 1. 为什么选 basic_2.yaml

| 证据位置 | 实际行为 |
| --- | --- |
| `run_ros.sh` | 启动现场安装目录 `~/hesai_test/install/share/pipeline_control/launch/run.launch` |
| `pipeline_control/launch/run.launch:8` | 唯一启用的 basic 配置加载项为 `basic_2.yaml`；basic_1、basic_3 已注释 |
| `pipeline_control/config/basic_2.yaml:1` | `car_id: "2"` |
| 归档根目录 `README.md` 的标定采集说明 | 将 basic_2 对应四川什邡，将 basic_3 对应上海；与实际主 launch 一致 |
| `pipeline_control/src/dense_feature_pub.cc:568` | 从全局 ROS 参数 `/scanner_z_to_robot_base/*` 读取七个数，构造 `kTbsz` |
| `pipeline_control/src/dense_feature_pub.cc:282` | 垛角点发送前实际执行 `kTbsz * corner_mat` |

主 launch 的 `car_index` 默认值虽然为 `3`，但该参数没有参与配置路径选择，不能据此认定加载了 basic_3。
已用 XML 解析器排除注释并递归检查其 include：scanner、motion_plan、localization、teaser_relocalization、line_relocalization、point_to_scan。
该启动链中未发现其他 basic 配置或 `out_calib.yaml` 覆盖上述标定参数。
另外单独运行的 `lidar_drive.launch`、`test_scan.launch`、部分标定 launch 会加载 basic_3；它们不属于上述主 launch 的 include 链。

## 2. 有效参数原值

以下平移单位均为米；四元数分量由键名确定。完整有效摘录见 [shifang_calibration_extracted.yaml](shifang_calibration_extracted.yaml)。

```yaml
scanner_z_to_robot_base:
  x: 0.629
  y: -0.469193
  z: -0.558983
  qx: 0.00271
  qy: 0.0007761
  qz: 0.0072169
  qw: 0.99997

lidar_to_scanner:
  x: 0
  y: 0.0
  z: 0.000261
  qx: 0.0
  qy: 0.0
  qz: 0.707106
  qw: 0.707106

scanner_z_to_scanner:
  roll: 0
  temp_angle_offset: 720
  scanner_lidar_rotation: 90

robot_base_to_base_link:
  x: 0.57
  y: 0.475
  z: 0.544
  qw: 1.0
  qx: 0.0
  qy: 0.0
  qz: 0.0
```

来源：`basic_2.yaml:4`、`:68`、`:98`、`:121`。
机械臂外参平移换成毫米为 `(629, -469.193, -558.983)`；底盘相对机械臂的平移为 `(570, 475, 544) mm`。
机械臂外参旋转按 `Rz(yaw) Ry(pitch) Rx(roll)` 表示，约为
`roll=0.3111775°、pitch=0.0866907°、yaw=0.8272417°`，这是四元数归一化后计算出的解释值。

配置中保留了多轮调整记录：X/Y 平移、roll/pitch/yaw 修正，以及“0818 yaw 调小 0.2 度”。
这些注释没有年份，不能推定具体年度。它们说明有效值已包含现场修正，不能用最前面的注释值替代。

## 3. 坐标含义和真正的数据流

统一记 `T_A_B` 为将 B 系坐标变换到 A 系，即 `p_A = T_A_B * p_B`，使用齐次列向量。

- `L_i`：扫描过程中某一时刻的主雷达原始坐标系，输入话题 `/xt32_1`。
- `S`：转台扫描参考坐标系。
- `Sz`：经 `scanner_z_to_scanner/roll` 反向旋转后的扫描参考系；本配置 roll=0，S 与 Sz 重合。
- `B`：FANUC 机械臂基座坐标系，对应 `robot_base`。
- `H`：AGV 底盘坐标系，对应 `base_link`。

```mermaid
flowchart LR
    L["原始雷达点 L_i /xt32_1"] -->|"T_Si_Li：lidar_to_scanner"| Si["瞬时转台系 S_i"]
    Si -->|"Rx(theta)：转台角度"| S["扫描参考系 S"]
    S -->|"Rx(-roll)，当前为单位阵"| Sz["扫描参考系 Sz：稠密点云及角点"]
    Sz -->|"T_B_Sz：scanner_z_to_robot_base"| B["机械臂基座 B"]
    B -->|"T_H_B：robot_base_to_base_link"| H["AGV 底盘 H"]
```

实际代码路径：

1. 主 launch 启动 `ext_dense_feature`、`dense_feature_pub` 和转台控制节点。
   `scanner/launch/scanner.launch:3` 中独立 `scan_node` 已注释；当前点云拼接由 `ext_dense_feature` 自己完成。
2. `ext_dense_feature/interface/ext_dense_feature.cpp:303` 读取 `lidar_to_scanner`，用 `Eigen::Quaterniond(qw,qx,qy,qz)` 构造 `Tsl_`；`:315` 订阅 `/xt32_1` 和 `/motor_angle`。
   launch 中虽声明了 `/dense_scan` 输入，实际订阅 `/dense_scan` 的代码已在 `:294` 至 `:300` 注释。
3. 该文件 `:108` 至 `:112` 按转台角构造绕 X 轴的 `Tssi`，计算 `Tsli = Tssi * Tsl_`。
   `thirdparty/motor/src/interp.cu:68` 至 `:109` 按点时间戳插值平移和旋转向量，对原始点执行 `R*p+t`。
   `ext_dense_feature.cpp:161` 至 `:164` 再施加 `Rx(-roll)`，得到 Sz 下的拼接点云。
4. `scanner/src/motor_control_new.cpp:34` 至 `:40` 发布 `angle + temp_angle_offset_`。
   当前 offset=720°，对几何旋转相当于两个整周；该消息还包含扫描控制标记，不能把所有消息数值当成实际旋转角。
   `scanner_lidar_rotation=90` 的读取位于标定节点 `scanner_lidar_calibration.cpp:232`，不是正常提取时额外乘一次 90°；90° 已体现在 `lidar_to_scanner` 四元数中。
5. `ext_dense_feature.cpp:422` 将拼接点云交给 Tracker；`interface/run_interface.hpp:268` 至 `:280` 发布角点位姿。
   消息字段仍叫 `T_lidar_marker`，但这里实际表示扫描参考系下的角点，不能按原始雷达系再乘一次 `lidar_to_scanner`。
   `/dense_scan` 的 frame_id 写作 `scanner`（`:388`），实际点数据已经过 Sz 变换；本配置 roll=0，所以两者一致。
6. `dense_feature_pub.cc:282` 将垛角点乘 `kTbsz`；`:292` 起写入输出字段，`:433` 交给 `SetFeatureMessage`。
   `pipeline_control/include/fanuc_pose/fanuc_pose.h:18` 把米乘 1000 转毫米，`:35` 起把旋转转成度制 W/P/R。
   `dense_feature_pub.cc:532` 创建的是端口 9002 的特征通信服务。

因此，原始雷达点到机械臂的几何链为：

```text
p_B = T_B_Sz * Rx(-roll) * Rx(theta) * T_Si_Li * p_Li
theta = /motor_angle 中的有效扫描角（度）
```

在扫描过程中 `theta` 随时间变化；代码对相邻姿态插值后作用到各点。因此不能用一个固定矩阵替代整段扫描的转台补偿。
若输入已经是该程序的 `/dense_scan`、保存的拼接点云或检测角点，则只需 `T_B_Sz`。

## 4. 可直接核对的矩阵

`dense_feature_pub` 对 YAML 四元数直接执行 Eigen 的 `toRotationMatrix()`，没有先显式归一化。
按这一实现计算的 `T_B_Sz` 为（平移列单位米）：

```text
 0.9998946280  -0.0144291605   0.0015912690   0.6290000000
 0.0144375734   0.9998811445  -0.0054086353  -0.4691930000
-0.0015130378   0.0054310395   0.9999841071  -0.5589830000
 0              0              0              1
```

例如输入点若以毫米表示，保持同一个 3×3 旋转，使用平移 `(629,-469.193,-558.983) mm` 即可。
方向是扫描系到机械臂系，不能把该矩阵直接用于反向转换。

仅当转台角为 0°（或整周等价角）、roll=0 时，原始雷达到机械臂可合成为
`T_B_L0 = T_B_Sz * T_S_L0`：

```text
-0.0144291605  -0.9998946280   0.0015912690   0.6290004153
 0.9998811445  -0.0144375734  -0.0054086353  -0.4691944117
 0.0054310395   0.0015130378   0.9999841071  -0.5587220041
 0              0              0              1
```

该合成中 `lidar_to_scanner` 按 `Sophus::SE3d(q,t)` 的归一化行为处理。
机械臂外参四元数范数约 `1.00000001548841`；雷达外参范数约 `0.99999889523539`。
归一化后的机械臂矩阵与直接 Eigen 矩阵最大元素差小于 `5e-10`，不影响上述数值的工程解释；若要求复现旧程序，应保留对这两条构造路径的区分。

## 5. 空车厢与垛角点输出不同

| 分支 | `dense_feature_pub.cc` 实际乘法 | 输出参考系 |
| --- | --- | --- |
| `ext_mode == 1`，垛角点 | `:282`，`kTbsz * corner_mat` | 机械臂基座 B |
| `ext_mode == 0`，空车厢角点 | `:221`，`kTbase_link_sz * corner_mat` | 底盘 H |

`:596` 定义 `kTbase_link_sz = Tbase_link_robot * kTbsz`。
本配置 `T_H_B` 无旋转，因此扫描系到 H 的旋转与 `T_B_Sz` 相同，平移为
`(1.199, 0.005807, -0.014983) m`。
空车厢分支 `:226` 的日志仍写着 “marker in fanuc”，不能用日志文字判断输出参考系。
后续 `:404` 起对导航目标再乘 180° 并将发布的 Z 设为 0，是导航目标构造；它没有回写之前已填入的角点报文。

## 6. 导航外参与历史数据的区别

导航也使用这组全局参数：

- `motion_plan/src/laser_scan_pub.cc:947`：主雷达到底盘为 `T_H_B * T_B_Sz * T_S_L0`。
- 同文件 `:966`：后雷达在上述结果右侧再乘 `vice_lidar_to_lidar_z`。
- `localization/thirdparty/HGO/include/lidars/lidar_options.h:43` 至 `:52` 使用同样的参数组合，原本从 `T_host_lidar` 数组读取的实现已注释。

后雷达相对零位主雷达的有效参数为：

```text
t = (-0.592878522787391, 2.57155261858015, -0.0213644875600896) m
q_xyzw = (0.999816777730975, -0.0152277946828371,
          0.00761185409471550, 0.00875128070562488)
```

下面这些数据不能替代本次确认的机械臂外参：

| 位置 | 核查结论 |
| --- | --- |
| `basic_2.yaml` 中注释掉的外参 | 历史标定/现场调试记录，不由 YAML 加载 |
| `basic_1.yaml` / `basic_3.yaml` | 其他配置，主运行链未选用 |
| `pipeline_control/config/out_calib.yaml` | 雷达与转台的自标定输出；主启动链未加载。输出来源可见 `scanner_lidar_calibration.cpp:195` 至 `:203` |
| 两份 `tf_compute.py` | 离线计算脚本，其 `(0.388330,-0.545112,-0.372473)` 等常数不在主运行链读取路径上 |
| `pipeline_control/plugin/scanner_base_calibration/config/Tbs.csv` | 离线标定程序输入；主启动链未读取 |
| `tracker0613.cpp`、`tracker_0610.cpp`、`tracker_0619_no_L.cpp`、`tracker_unrobust.cpp` | 备份代码；CMake 的当前目标明确编译 `src/tracker/tracker.cpp` |
| `pipeline_control/launch/run.launch:15` 静态 TF | 仍会广播 `(1.13762,0.325643,-0.327426)` 等硬编码值，但它是 `base_link` 到 `xt32_1` 的另一条 TF 定义；角点发送代码并未查询这条 TF |

该静态 TF 与从 basic_2 组合得到的主雷达到底盘结果不一致。后者零位平移约为
`(1.1990004153,0.0058055883,-0.0147220041) m`，旋转也不同。
因此不能把 TF 显示中的这条静态变换当成垛角点实际使用的机械臂标定。

`tracker.cpp:25` 至 `:46` 硬编码的机械臂外参和底盘外参均与 basic_2 有效值一致。
已追踪其使用点：`:1493`、`:1648`、`:2063` 等处用于输出调试坐标，没有把这些变换后的点重新写回角点消息。
真正发出的角点仍由 `dense_feature_pub` 读取 ROS 参数后换算；未来只改 YAML 时，Tracker 的硬编码日志可能失真。

## 7. 核查结果与确认边界

已完成：读取主启动链；按 XML 排除注释并递归检查配置覆盖；核对编译目标；跟踪点云、转台角度、角点消息与发送分支；逐字段比对 YAML 和 Tracker 七个外参数；计算矩阵及其组合；验证导出的 YAML 与原文件有效值一致。

未进行 ROS1/CUDA 编译或硬件联调。本归档没有提供现场安装目录的完整运行快照。
`run_ros.sh` 指向 install 目录，因此只检查源码目录不足以证明现场已部署同一套配置。
如需进一步核实在机版本，可在现场已配置 ROS 环境的终端只读查询：

```bash
rospack find pipeline_control
rosparam get /car_id
rosparam get /scanner_z_to_robot_base
rosparam get /lidar_to_scanner
rosparam get /scanner_z_to_scanner
rosparam get /robot_base_to_base_link
```

相关节点在初始化时缓存标定；参数服务器当前值并不自动证明已经运行的节点采用了这些值。
在机确认还应对照启动时配置、实际安装文件和进程启动时间。
对于后续独立角点检测程序，首先确认输入是原始 `/xt32_1`，还是旧程序已经补偿过转台角的拼接点云，再选择本报告中的相应变换链。

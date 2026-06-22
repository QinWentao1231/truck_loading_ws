"""一键启动：车厢宽度检测节点 + RViz。

可用 launch 参数覆盖关键项（也可继续用 config.json 默认值）：
  station_interval_m  站间位移(米)
  max_spread_mm       变形阈值(毫米)
  slice_step_m        切片步长(米)
  offline_pcd_dir     离线回放目录(留空=在线分站采集)
  use_rviz            是否同时起 RViz (true/false)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('cabin_width_check')
    rviz_cfg = os.path.join(share, 'rviz', 'cabin_width.rviz')

    args = [
        DeclareLaunchArgument('station_interval_m', default_value='3.0'),
        DeclareLaunchArgument('max_spread_mm',      default_value='60.0'),
        DeclareLaunchArgument('slice_step_m',       default_value='0.5'),
        DeclareLaunchArgument('offline_pcd_dir',    default_value=''),
        DeclareLaunchArgument('use_rviz',           default_value='true'),
    ]

    width_node = Node(
        package='cabin_width_check',
        executable='width_check_node',
        name='cabin_width_check',
        output='screen',
        parameters=[{
            'station_interval_m': LaunchConfiguration('station_interval_m'),
            'max_spread_mm':      LaunchConfiguration('max_spread_mm'),
            'slice_step_m':       LaunchConfiguration('slice_step_m'),
            'offline_pcd_dir':    LaunchConfiguration('offline_pcd_dir'),
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription(args + [width_node, rviz_node])

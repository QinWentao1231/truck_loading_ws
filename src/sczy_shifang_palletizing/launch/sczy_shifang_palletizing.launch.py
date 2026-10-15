"""启动独立ROS2节点；enable_network:=false可验证部署而不监听机器人端口。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_default = PathJoinSubstitution([
        FindPackageShare('sczy_shifang_palletizing'), 'config', 'planner.json'])
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=config_default),
        DeclareLaunchArgument('enable_network', default_value='true'),
        DeclareLaunchArgument('offline_order_file', default_value=''),
        DeclareLaunchArgument('log_dir', default_value=''),
        Node(
            package='sczy_shifang_palletizing',
            executable='sczy_shifang_palletizing_node',
            name='sczy_shifang_palletizing',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'enable_network': ParameterValue(
                    LaunchConfiguration('enable_network'), value_type=bool),
                'offline_order_file': ParameterValue(
                    LaunchConfiguration('offline_order_file'), value_type=str),
                'log_dir': ParameterValue(LaunchConfiguration('log_dir'), value_type=str),
            }],
        ),
    ])

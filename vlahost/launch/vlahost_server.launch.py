from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    host_arg = DeclareLaunchArgument(
        'host',
        default_value='0.0.0.0',
        description='Bind address for the vlahost HTTP server',
    )
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='8000',
        description='Bind port for the vlahost HTTP server',
    )

    vlahost_server_node = Node(
        package='vlahost',
        executable='vlahost_server',
        name='vlahost_server',
        output='screen',
        arguments=[
            '--host', LaunchConfiguration('host'),
            '--port', LaunchConfiguration('port'),
        ],
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        vlahost_server_node,
    ])

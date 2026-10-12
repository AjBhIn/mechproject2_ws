import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('turtlebot3_gazebo')
    urdf_path = os.path.join(pkg_dir, 'models', 'turtlebot3_burger_cam', 'turtlebot3_enemy_bot.sdf')
    bridge_params = os.path.join(pkg_dir, 'params', 'turtlebot3_enemy_bot_bridge.yaml')

    spawner = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'enemy_bot',
            '-file', urdf_path,
            '-x', '2.0',
            '-y', '0.0',
            '-z', '0.15',
            '-Y', '3.14',
            '-yaw', '3.14'
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen'
    )

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/enemy_bot/camera/image_raw'],
        output='screen'
    )

    return LaunchDescription([
        spawner,
        bridge,
        # image_bridge
    ])
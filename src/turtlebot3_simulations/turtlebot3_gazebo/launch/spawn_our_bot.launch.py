import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('turtlebot3_gazebo')
    urdf_path = os.path.join(pkg_dir, 'models', 'turtlebot3_burger_cam', 'turtlebot3_our_bot.sdf')
    bridge_params = os.path.join(pkg_dir, 'params', 'turtlebot3_burger_cam_bridge.yaml')

    spawner = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'our_bot',
            '-file', urdf_path,
            '-x', '-2.0',
            '-y', '0.0',
            '-z', '0.15'
        ],
        output='screen'
    )

    # Note: No namespace or remappings here because absolute paths are defined in bridge YAML[cite: 7]
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen'
    )

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/our_bot/camera/image_raw'],
        output='screen'
    )

    return LaunchDescription([
        spawner,
        bridge,
        # image_bridge
    ])
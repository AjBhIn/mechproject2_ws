import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('turtlebot3_gazebo')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'turtlebot3_enemy_bot.urdf')

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='enemy_bot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'robot_description': robot_desc
            }],
            remappings=[
                ('joint_states', '/enemy_bot/joint_states'),
                ('clock', '/clock')
            ]
        )
    ])
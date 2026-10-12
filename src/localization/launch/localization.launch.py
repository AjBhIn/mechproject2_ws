#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_localization = get_package_share_directory('localization')

    map_file = os.path.expanduser('~/mechproject2_ws/src/map_server/my_nav_map.yaml')
    default_rviz_path = os.path.expanduser('~/mechproject2_ws/rvizfiles/rvizfortwobot.rviz')
    
    rviz_config_file = LaunchConfiguration('rviz_config')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_path, 
        description='Full path to the RViz config file to use'
    )

    # 1. SHARED MAP SERVER & RVIZ
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': True, 'yaml_filename': map_file}]
    )

    map_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': True, 
            'autostart': True, 
            'node_names': ['map_server']
        }]
    )

    # 2. OUR_BOT AMCL
    our_bot_amcl = Node(
        namespace='our_bot',
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            os.path.join(pkg_localization, 'config', 'our_bot_amcl.yaml'),
            {'use_sim_time': True}
        ],
        remappings=[('map', '/map')]
    )

    lifecycle_our_bot_amcl = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_amcl_our_bot',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': ['our_bot/amcl']
        }]
    )

    # 3. ENEMY_BOT AMCL
    enemy_bot_amcl = Node(
        namespace='enemy_bot',
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            os.path.join(pkg_localization, 'config', 'enemy_bot_amcl.yaml'),
            {'use_sim_time': True}
        ],
        remappings=[('map', '/map')]
    )

    lifecycle_enemy_bot_amcl = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_amcl_enemy_bot',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': ['enemy_bot/amcl']
        }]
    )

    ld = LaunchDescription()
    ld.add_action(declare_rviz_config_cmd)
    ld.add_action(rviz_node)
    ld.add_action(map_server_node)
    ld.add_action(map_lifecycle_node)

    ld.add_action(our_bot_amcl)
    ld.add_action(lifecycle_our_bot_amcl)

    ld.add_action(enemy_bot_amcl)
    ld.add_action(lifecycle_enemy_bot_amcl)

    return ld
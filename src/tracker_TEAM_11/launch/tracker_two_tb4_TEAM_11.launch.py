#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_namespace = LaunchConfiguration('chaser')
    enemy_namespace = LaunchConfiguration('enemy')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        # -------------------------------
        # CHANGE THIS FOR PHYSICAL BOT TO FALSE
        #--------------------------------
        default_value='true',
        description='Use simulation (Gazebo) clock if true, hardware clock if false'
    )

    declare_robot_namespace_cmd = DeclareLaunchArgument(
        'chaser',
        default_value='our_bot',
        description='Namespace of our robot'
    )

    declare_enemy_namespace_cmd = DeclareLaunchArgument(
        'enemy',
        default_value='enemy_bot',
        description='Namespace of the enemy robot'
    )

    # Centralized parameters guaranteed to be passed to EVERY node
    common_tracking_params = {
        'use_sim_time': use_sim_time,
        'chaser': robot_namespace,
        'enemy': enemy_namespace,
    }

    # Kill Pose (The Carrot)
    kill_pose = Node(
        package='tracker_TEAM_11', 
        executable='kill_pose',
        name='kill_pose',
        # namespace=robot_namespace,
        parameters=[common_tracking_params],
        output='screen'
    )

    # Evasion Calculator (The Strategist)
    evasion_calculator = Node(
        package='tracker_TEAM_11', 
        executable='evasion_calculator',
        name='evasion_calculator',
        # namespace=robot_namespace,
        parameters=[common_tracking_params],
        output='screen'
    )

    # Chaser Calculator
    chaser_calculator = Node(
        package='tracker_TEAM_11', 
        executable='chaser_calculator',
        name='chaser_calculator',
        # namespace=robot_namespace,
        parameters=[common_tracking_params],
        output='screen'
    )

    # State machine 
    state_machine = Node(
        package='tracker_TEAM_11', 
        executable='state_machine',
        name='state_machine',
        # namespace=robot_namespace,
        parameters=[common_tracking_params],
        output='screen'
    )

    # Goal Sender 
    goal_sender = Node(
        package='tracker_TEAM_11', 
        executable='goal_sender',
        name='goal_sender',
        # namespace=robot_namespace,
        parameters=[common_tracking_params],
        output='screen'
    )

    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_robot_namespace_cmd)
    ld.add_action(declare_enemy_namespace_cmd)
    
    ld.add_action(kill_pose)
    ld.add_action(evasion_calculator)
    ld.add_action(chaser_calculator)
    ld.add_action(state_machine)
    ld.add_action(goal_sender)

    return ld
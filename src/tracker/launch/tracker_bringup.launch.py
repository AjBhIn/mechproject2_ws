#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true, hardware clock if false'
    )

    # Kill Pose (The Carrot)
    kill_pose = Node(
        package='tracker', 
        executable='kill_pose',
        name='kill_pose',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Evasion Calculator (The Strategist)
    evasion_calculator = Node(
        package='tracker', 
        executable='evasion_calculator',
        name='evasion_calculator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Goal Sender 
    goal_sender = Node(
        package='tracker', 
        executable='goal_sender',
        name='goal_sender',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Goal Sender 
    chaser_calculator = Node(
        package='tracker', 
        executable='chaser_calculator',
        name='goal_sender',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )


    # State machine 
    state_machine = Node(
        package='tracker', 
        executable='state_machine',
        name='state_machine',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time_cmd)
    
    ld.add_action(kill_pose)
    ld.add_action(evasion_calculator)
    ld.add_action(chaser_calculator)
    ld.add_action(state_machine)
    ld.add_action(goal_sender)

    return ld
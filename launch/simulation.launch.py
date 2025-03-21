import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    drl_nav_path = get_package_share_directory("drl_navigation")
    gazebo_ros_path = get_package_share_directory("gazebo_ros")
    world_file_path = os.path.join(drl_nav_path, "worlds/TD3.world")

    launch_actions = []

    gui_arg = DeclareLaunchArgument(
        "gui", default_value="true", description="Launch gazebo client"
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true", description="Launch rviz"
    )
    world_arg = DeclareLaunchArgument(
        "world_name", default_value=world_file_path, description="World file"
    )
    gz_verbose_arg = DeclareLaunchArgument(
        "gz_verbose", default_value="true", description="Enable verbose server output"
    )

    # Include Gazebo server and client launch files
    gz_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_path, "launch", "gzserver.launch.py")
        ),
        launch_arguments={
            "world": LaunchConfiguration("world_name"),
            "verbose": LaunchConfiguration("gz_verbose"),
        }.items(),
    )

    gz_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_path, "launch", "gzclient.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("gui")),
        launch_arguments={"verbose": LaunchConfiguration("gz_verbose")}.items(),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(drl_nav_path, "launch", "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    spawn_entity_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(drl_nav_path, "launch", "spawn_entity.launch.py")
        ),
        launch_arguments={
            "x": "0.0",
            "y": "0.0",
            "z": "0.01",
            "yaw": "0.0",
        }.items(),
    )

    rviz_cmd = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(drl_nav_path, "rviz/tb3_navigation.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    launch_actions = [
        gui_arg,
        rviz_arg,
        world_arg,
        gz_verbose_arg,
        gz_server_cmd,
        gz_client_cmd,
        robot_state_publisher_cmd,
        spawn_entity_cmd,
        rviz_cmd,
    ]
    return LaunchDescription(launch_actions)

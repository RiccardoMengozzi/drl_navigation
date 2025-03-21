import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_actions = []
    # Get the urdf file
    TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]
    model_folder = "turtlebot3_" + TURTLEBOT3_MODEL
    urdf_path = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"),
        "models",
        model_folder,
        "model.sdf",
    )

    # Launch configuration variables specific to simulation
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")
    yaw_pose = LaunchConfiguration("yaw_pose", default="0.0")

    # Declare the launch arguments
    x_pose_arg = DeclareLaunchArgument(
        "x_pose", default_value="0.0", description="Specify namespace of the robot"
    )

    y_pose_arg = DeclareLaunchArgument(
        "y_pose", default_value="0.0", description="Specify namespace of the robot"
    )

    yaw_pose_arg = DeclareLaunchArgument(
        "yaw_pose", default_value="0.0", description="Specify namespace of the robot"
    )

    spawn_entity_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", TURTLEBOT3_MODEL,
            "-file", urdf_path,
            "-x", x_pose,
            "-y", y_pose,
            "-z", "0.01",
            "-Y", yaw_pose,
        ],
        output="screen",
    )

    launch_actions = [
        x_pose_arg,
        y_pose_arg,
        yaw_pose_arg,
        spawn_entity_cmd,
    ]

    return LaunchDescription(launch_actions)


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    launch_actions = []

    TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    urdf_file_name = "turtlebot3_" + TURTLEBOT3_MODEL + ".urdf"
    frame_prefix = LaunchConfiguration("frame_prefix", default="")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    frame_prefix_arg = DeclareLaunchArgument(
        "frame_prefix",
        default_value="",
        description="Prefix to be added to the tf frame names",
    )

    urdf_path = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "urdf", urdf_file_name
    )

    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()

    robot_state_pub_cmd = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": robot_desc,
                "frame_prefix": PythonExpression(["'", frame_prefix, "/'"]),
            }
        ],
    )

    launch_actions = [use_sim_time_arg, frame_prefix_arg, robot_state_pub_cmd]

    return LaunchDescription(launch_actions)

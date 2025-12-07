from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    bounce = Node(
        package="project",
        executable="ball_marker",
        name="ball_marker"
    )

    return LaunchDescription([bounce])

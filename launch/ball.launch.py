"""
Launch a single ball above the Panda arm using its URDF.

- Loads ball.urdf with robot_state_publisher
- Positions it slightly above the Panda base
- Publishes the ball TF relative to panda_0
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # --- locate ball.urdf ---
    pkg_share = get_package_share_directory('project')
    ball_urdf_file = os.path.join(pkg_share, 'urdf', 'ball.urdf')

    if not os.path.exists(ball_urdf_file):
        raise FileNotFoundError(f"Cannot find {ball_urdf_file}")

    with open(ball_urdf_file, 'r') as f:
        ball_urdf = f.read()

    # --- position above Panda base ---
    x = 0.0
    y = 0.0
    z = 0.3  # slightly above the robot

    # --- Node to publish the URDF ---
    ball_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ball_state_publisher',
        output='screen',
        parameters=[{'robot_description': ball_urdf}]
    )

    # --- Node to place it above Panda using a static transform ---
    ball_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ball_tf_publisher',
        arguments=[
            '0', '0', '1.5',   # translation
            '0', '0', '0',            # rotation (roll, pitch, yaw)
            'panda_link0',                # parent frame
            'ball_link'               # child frame (root link of ball.urdf)
        ]
    )

    return LaunchDescription([ball_node, ball_tf_node])
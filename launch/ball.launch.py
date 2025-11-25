import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_share = get_package_share_directory('project')
    ball_urdf_file = os.path.join(pkg_share, 'urdf', 'ball.urdf')

    if not os.path.exists(ball_urdf_file):
        raise FileNotFoundError(f"Cannot find {ball_urdf_file}")

    with open(ball_urdf_file, 'r') as f:
        ball_urdf = f.read()

    x = 0.0
    y = 0.0
    z = 0.3  # slightly above the robot

    ball_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ball_state_publisher',
        output='screen',
        parameters=[{'robot_description': ball_urdf}]
    )

    ball_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ball_tf_publisher',
        arguments=[
            '0', '0', '1.5',   # translation
            '0', '0', '0',       # rotation 
            'panda_link0',                # parent frame
            'ball_link'               # child frame 
        ]
    )

    return LaunchDescription([ball_node, ball_tf_node])

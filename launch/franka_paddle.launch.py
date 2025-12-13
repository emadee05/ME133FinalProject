import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_share = get_package_share_directory('project')
    franka_urdf_file = os.path.join(pkg_share, 'urdf', 'franka_paddle.urdf')

    if not os.path.exists(franka_urdf_file):
        raise FileNotFoundError(f"Cannot find {franka_urdf_file}")

    with open(franka_urdf_file, 'r') as f:
        franka_urdf = f.read()

    # This is the *main* robot_state_publisher the KinematicChain will use
    franka_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': franka_urdf}],
    )

    return LaunchDescription([franka_node])

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import Shutdown


def launch_main_urdf(context, *args, **kwargs):

    urdf = LaunchConfiguration('urdf').perform(context)
    package = 'project'

    rvizcfg = os.path.join(pkgdir(package), 'rviz/viewurdf.rviz')

    if urdf == '':
        print("Please specify a URDF with: urdf:=<FILENAME>")
        return []
    if not os.path.exists(urdf):
        print(f"URDF file '{urdf}' does not exist")
        return []

    with open(urdf, 'r') as f:
        robot_description = f.read()

    node_rsp = Node(
        name='robot_state_publisher',
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    node_rviz = Node(
        name='rviz',
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rvizcfg],
        on_exit=Shutdown()
    )

    # GUI
    node_gui = Node(
        name='joint_state_publisher_gui',
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        on_exit=Shutdown()
    )

    return [node_rsp, node_rviz, node_gui]


def launch_ball():

    pkg_share = pkgdir('project')
    ball_urdf_file = os.path.join(pkg_share, 'urdf', 'ball.urdf')

    if not os.path.exists(ball_urdf_file):
        raise FileNotFoundError(f"Cannot find {ball_urdf_file}")

    with open(ball_urdf_file, 'r') as f:
        ball_urdf = f.read()

    ball_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ball_state_publisher',
        output='screen',
        parameters=[{'robot_description': ball_urdf}]
    )

    ball_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ball_tf_publisher',
        arguments=[
            '0', '0', '2',        # translation
            '0', '0', '0',        # roll pitch yaw
            'panda_link0',        # parent frame
            'ball_link'           # child frame
        ]
    )

    return [ball_node, ball_tf]

def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument('urdf', default_value=''),

        OpaqueFunction(function=launch_main_urdf),

        *launch_ball()
    ])

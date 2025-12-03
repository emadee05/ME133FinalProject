import rclpy
import numpy as np
import tf2_ros

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from .KinematicChain import KinematicChain
from .TrajectoryUtils import *
from .TransformHelpers import *

class TrajectoryNode(Node):
    def __init__(self, name="trajectory"):
        super().__init__(name)

        # Joint names
        self.jointnames = [
            'panda_joint1','panda_joint2','panda_joint3',
            'panda_joint4','panda_joint5','panda_joint6','panda_joint7'
        ]

        # Kinematic chain
        self.chain = KinematicChain(self, 'panda_link0', 'panda_paddle', self.jointnames)

        # Initial joint position
        self.qc = np.radians(np.array([0, 90, 0, -90, 0, 0, 0]))
        (self.p0, self.R0, _, _) = self.chain.fkin(self.qc)
        self.qcdot = np.zeros(7)

        # Hover Z position slightly above 0
        self.hover_z = max(self.p0[2], 0.05)

        # Timer settings
        self.dt = 0.01  # 100Hz
        self.now = self.get_clock().now()

        # Publishers
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while not self.count_subscribers('/joint_states'):
            pass

        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info(f"Running hover node at {1/self.dt:.1f} Hz")

    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()

    def update(self):
        # Use current time
        now = self.get_clock().now()

        # Desired pose
        pd = self.p0.copy()
        pd[2] = max(self.p0[2], 0.05)
        Rd = self.R0.copy()

        # Forward kinematics
        _, _, Jv, Jw = self.chain.fkin(self.qc)
        J = np.vstack((Jv, Jw))
        pc, Rc, _, _ = self.chain.fkin(self.qc)

        # PD control
        ep = pd - pc
        eRnow = eR(Rd, Rc)
        xdot = np.concatenate((ep / self.dt, eRnow / self.dt))
        self.qcdot = np.linalg.pinv(J) @ xdot
        self.qc = self.qc + self.qcdot * self.dt

        # Header
        header = Header(stamp=now.to_msg(), frame_id='world')

        # Publish joint state
        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=self.qc.tolist(),
            velocity=self.qcdot.tolist()))

        # Publish pose
        self.pubpose.publish(PoseStamped(header=header, pose=Pose_from_Rp(Rd, pd)))

        # Publish zero twist
        self.pubtwist.publish(TwistStamped(header=header, twist=Twist_from_vw(vzero(), vzero())))

        # Broadcast TF
        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id='desired',
            transform=Transform_from_Rp(Rd, pd)))


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)  # keep node running
    except KeyboardInterrupt:
        node.get_logger().info("Hover node stopped by user")
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

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

        self.chain = KinematicChain(self, 'panda_link0', 'panda_paddle', self.jointnames)

        # slight tilt so we can see trajectory
        self.qc = np.radians(np.array([0, 90, 0, -90, 0, -30, 0]))
        (self.p0, self.R0, _, _) = self.chain.fkin(self.qc)
        self.qcdot = np.zeros(7)

        # Hover Z position slightly above 0
        self.hover_z = max(self.p0[2], 0.05)

        # Oscillation
        self.amp = 0.05  
        self.freq = 0.5  
        self.start_time = self.get_clock().now().nanoseconds * 1e-9  # ns to seconds

        self.dt = 0.01  # 100Hz
        self.now = self.get_clock().now()

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
        # Current time
        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        # Desired pose with vertical oscillation
        pd = self.p0.copy()
        pd[2] = self.hover_z + self.amp * np.sin(2 * np.pi * self.freq * t)
        Rd = self.R0.copy()

        _, _, Jv, Jw = self.chain.fkin(self.qc)
        J = np.vstack((Jv, Jw))
        pc, Rc, _, _ = self.chain.fkin(self.qc)

        ep = pd - pc
        eRnow = eR(Rd, Rc)
        xdot = np.concatenate((ep / self.dt, eRnow / self.dt))
        self.qcdot = np.linalg.pinv(J) @ xdot
        self.qc = self.qc + self.qcdot * self.dt

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')

        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=self.qc.tolist(),
            velocity=self.qcdot.tolist()))

        self.pubpose.publish(PoseStamped(header=header, pose=Pose_from_Rp(Rd, pd)))

        self.pubtwist.publish(TwistStamped(header=header, twist=Twist_from_vw(vzero(), vzero())))

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

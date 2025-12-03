import rclpy
import numpy as np
import tf2_ros

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header

from .KinematicChain import KinematicChain
from .TrajectoryUtils import *
from .TransformHelpers import *

class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']
        self.jointnames = names

        self.chain = KinematicChain(self, 'panda_link0', 'panda_paddle', self.jointnames)

        self.q0 = np.radians(np.array([0, 90, 0, -90, 0, 0, 0]))
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)

        self.pflat = self.p0 + np.array([0.5, 0.0, 0.0])  # 0.5m forward
        self.Rflat = Rotx(-np.pi/2) @ Roty(-np.pi/2)       # horizontal orientation

        self.qc = self.q0.copy()
        self.ep = vzero()
        self.eR = vzero()
        self.lam = 20

        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        self.dt    = 0.01                       # 100Hz
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))

    # Shutdown
    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()

    # Update function
    def update(self):
        self.t   = self.t + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)

        # Linear interpolation to flat target over 3 seconds
        total_time = 3.0
        s0 = min(self.t / total_time, 1.0)       
        s0dot = 1.0 / total_time                 # constant velocity

        pd = self.p0 + (self.pflat - self.p0) * s0
        vd = (self.pflat - self.p0) * s0dot

        Rd = self.Rflat
        wd = np.zeros(3)

        (_, _, Jv, Jw) = self.chain.fkin(self.qc)
        J = np.vstack((Jv, Jw))

        linearv = vd + (0.1/self.dt) * self.ep
        angularv = wd + (0.1/self.dt) * self.eR

        xdot = np.concatenate((linearv, angularv))
        self.qcdot = np.linalg.pinv(J) @ xdot
        self.qc = self.qc + self.qcdot * self.dt

        (pc, Rc, Jv, Jw) = self.chain.fkin(self.qc)
        self.ep = pd - pc
        self.eR = eR(Rd, Rc)

        qc = self.qc
        qcdot = self.qcdot

        self.L = 0.4
        diagonal = np.diag([1/self.L, 1/self.L, 1/self.L, 1.0, 1.0, 1.0])
        J_bar = diagonal @ J

        header = Header(stamp=self.now.to_msg(), frame_id='world')
        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()))
        self.pubpose.publish(PoseStamped(
            header=header,
            pose=Pose_from_Rp(Rd,pd)))
        self.pubtwist.publish(TwistStamped(
            header=header,
            twist=Twist_from_vw(vd,wd)))
        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id='desired',
            transform=Transform_from_Rp(Rd,pd)))

def main(args=None):
    rclpy.init(args=args)
    future = Future()
    trajectory = TrajectoryNode('trajectory_flat', future)
    rclpy.spin_until_future_complete(trajectory, future)
    trajectory.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

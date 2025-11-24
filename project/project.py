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
from. TransformHelpers import *

class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        # Define the list of joint names MATCHING THE JOINT NAMES IN THE URDF!
        names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']
        self.jointnames = names

        # Set up the kinematic chain object.
        self.chain = KinematicChain(self, 'panda_link0', 'panda_paddle', self.jointnames)


        # Define the matching initial joint/task positions.
        self.q0 = np.radians(np.array([0, 90,   0,   -90, 0, 0, 0]))
        #self.p0 = np.array([0.0, 0.55, 1.0])
        #self.R0 = Reye()
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)

        # Define the other points.
        self.pleft  = np.array([ 0.3, 0.5, 0.15])
        self.pright = np.array([-0.3, 0.5, 0.15])
        self.Rleft  = Rotx(-np.pi/2) @ Roty(-np.pi/2)
        self.Rleft  = Rotz( np.pi/2) @ Rotx(-np.pi/2)
        self.Rright = Reye()

        # Initialize the stored joint command position and task errors.
        self.qc = self.q0.copy()
        self.ep = vzero()
        self.eR = vzero()

        # Pick the convergence bandwidth.
        self.lam = 20



        ##############################################################
        # Setup the logistics of the node:
        # Add publishers to send the joint and task commands.  Also
        # add a TF broadcaster, so the desired pose appears in RVIZ.
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Set up the timer to update at 100Hz, with (t=0) occuring in
        # the first update cycle (dt) from now.
        self.dt    = 0.01                       # 100Hz.
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time since 1970
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()


    def update(self):
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt) 
        # COMPUTE THE TRAJECTORY AT THIS TIME INSTANCE.
        (s0, s0dot) = goto(self.t, 3.0, 0.0, 1.0)

        pd = self.p0 + (self.pright - self.p0) * s0
        vd =           (self.pright - self.p0) * s0dot

        Rd = Reye()
        wd = np.zeros(3)

        # KINEMATICS
        (_, _, Jv, Jw) = self.chain.fkin(self.qc)
        J = np.vstack((Jv, Jw))
        # get vr and wr from the vd + lamda(error concatenated) - equation in notes
        linearv = vd + (0.1/self.dt) * self.ep
        angularv = wd + (0.1/self.dt) * self.eR

        xdot = np.concatenate((linearv, angularv))
        self.qcdot = np.linalg.pinv(J) @ xdot
        self.qc = self.qc + self.qcdot * self.dt
        # run in again
        (pc, Rc, Jv, Jw) = self.chain.fkin(self.qc)
        self.ep = pd - pc
        self.eR = eR(Rd, Rc)

        qc = self.qc
        qcdot = self.qcdot
        self.L = 0.4
        diagonal = np.diag([1/self.L, 1/self.L, 1/self.L, 1.0, 1.0, 1.0])
        J_bar = diagonal @ J
        

        header=Header(stamp=self.now.to_msg(), frame_id='world')
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
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Initialize the trajectory generator node.
    trajectory = TrajectoryNode('trajectory', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory is
    # complete (as signaled by the future object).
    rclpy.spin_until_future_complete(trajectory, future)

    # Report the reason for shutting down.
    if future.done():
        trajectory.get_logger().info("Stopping: " + future.result())
    else:
        trajectory.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    trajectory.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

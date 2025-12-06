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
from visualization_msgs.msg import Marker
from geometry_msgs.msg  import PoseStamped, TwistStamped, PointStamped


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
        # names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7', 'panda_joint8']
        names = [
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ]

        self.jointnames = names

        # Set up the kinematic chain object.
        self.chain = KinematicChain(self, 'panda_link0', 'panda_paddle', self.jointnames)

        # Define the matching initial joint/task positions.        
        self.q0 = (np.array([0, 0.7, -0.3, -1.5, 0.3, 1.9, -1.2]))
        self.qgoal = np.radians([45, 60, 10, -120, 0, 10, 0])
        self.T = 15.0

        # === task-space goal
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)
        self.goal_p = self.p0 + np.array([0.1, 0.0, 0.1])  
        self.goal_R = self.R0                        

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

        self.ball_pos = None   # np.array([x, y, z])
        self.ball_vel = None   # np.array([vx, vy, vz])
        self.have_plan = False
        self.initial_height = 1.0      # must match ball node
        self.reset_eps_z = 0.02        # tolerance on height
        self.reset_eps_v = 0.05        # tolerance on velocity
        self.g = -9.81         # gravity

        self.have_plan = False     # whether we've computed an intercept
        self.p_start = self.p0     # start pose for spline
        self.R_start = self.R0

        # Subscribers for ball position and velocity
        self.ball_pos_sub = self.create_subscription(
            PointStamped,
            '/ball_position',          # <-- change if your topic name is different
            self.ball_pos_callback,
            10
        )

        self.ball_vel_sub = self.create_subscription(
            TwistStamped,
            '/ball_velocity',          # <-- change if your topic name is different
            self.ball_vel_callback,
            10
        )

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

    def ik_to_joints(self, pd, Rd=None, qi = None, dt=0.01, c=0.5, max_iters=200, tol=1e-4):
        '''
        given desired position/orientation, integrate qdot = pinv(J) * xdot until we reach pose and we get q
        returns the q and if it can reach
        '''
        if qi is None:
            q = self.q0.copy()
        else:
            q = qi.copy()
        for k in range(max_iters):
            pc, Rc, Jv, Jw = self.chain.fkin(q)
            err = pd - pc
            if np.linalg.norm(err)<tol:
                return q, True
            # jac for pos, we care about the cartesian position
            J = Jv 
            dq = c*(np.linalg.pinv(J)@err)
            q = q+dq
        return q, False
        # J = np.vstack((Jv, Jw))
        # # get vr and wr from the vd + lamda(error concatenated) - equation in notes
        # linearv = vd + (0.1/self.dt) * self.ep
        # angularv = wd + (0.1/self.dt) * self.eR

        # xdot = np.concatenate((linearv, angularv))
        # self.qcdot = np.linalg.pinv(J) @ xdot
        # self.qc = self.qc + self.qcdot * self.dt
        # # run in again
        # (pc, Rc, Jv, Jw) = self.chain.fkin(self.qc)
        # self.ep = pd - pc
        # self.eR = eR(Rd, Rc)

        # qc = self.qc
        # qcdot = self.qcdot
        # self.L = 0.4
        # diagonal = np.diag([1/self.L, 1/self.L, 1/self.L, 1.0, 1.0, 1.0])
        # J_bar = diagonal @ J

    def ball_pos_callback(self, msg: PointStamped):
        # Assuming msg.header.frame_id == 'panda_link0'
        self.ball_pos = np.array([
            msg.point.x,
            msg.point.y,
            msg.point.z
        ])

    def ball_vel_callback(self, msg: TwistStamped):
        # Assuming linear velocity in the same frame as ball_pos
        self.ball_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])
    def update(self):
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt) 
        if self.ball_pos is not None and self.ball_vel is not None:
            z  = self.ball_pos[2]
            vz = self.ball_vel[2]

            # Ball is back near spawn height and almost stationary -> treat as a new trial
            if (abs(z - self.initial_height) < self.reset_eps_z
                    and np.linalg.norm(self.ball_vel) < self.reset_eps_v):
                if self.have_plan:
                    self.get_logger().info("Ball respawned, clearing old plan.")
                self.have_plan = False
        if (not self.have_plan) and (self.ball_pos is not None) and (self.ball_vel is not None):
            z0  = self.ball_pos[2]
            vz0 = self.ball_vel[2]

            # Use current paddle height as target hit height
            pc_current, Rc_current, _, _ = self.chain.fkin(self.qc)
            z_hit = pc_current[2]  # you can add ball radius here if needed

            # Only plan if ball is above paddle and moving downward
            if (z0 > z_hit) and (vz0 < 0.0):
                a = 0.5 * self.g
                b = vz0
                c = z0 - z_hit

                disc = b*b - 4*a*c
                if disc > 0.0:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-b + sqrt_disc) / (2*a)
                    t2 = (-b - sqrt_disc) / (2*a)
                    t_candidates = [t for t in (t1, t2) if t > 0.0]

                    if t_candidates:
                        t_hit = min(t_candidates)

                        # Plan a task-space spline from current paddle pose
                        self.T = float(t_hit)
                        self.t = 0.0  # reset internal time

                        self.p_start, self.R_start, _, _ = self.chain.fkin(self.qc)

                        x_hit = self.ball_pos[0]
                        y_hit = self.ball_pos[1]
                        self.goal_p = np.array([x_hit, y_hit, z_hit])
                        self.qgoal, self.reach = self.ik_to_joints(self.goal_p) # also put desired orientation of tip
                        self.goal_R = self.R_start  # keep current orientation

                        self.have_plan = True
                        self.get_logger().info(
                            f"Planned intercept: T={self.T:.3f}s, goal_p={self.goal_p}"
                        )
        # COMPUTE THE TRAJECTORY AT THIS TIME INSTANCE.
        
        # # === q0 to goal via joint spline 
        if self.t >= self.T:
            # self.t = self.T

            # ===== HOLD FINAL JOINT POSITION =====
            qc    = self.qgoal.copy()
            qcdot = np.zeros_like(self.qgoal)
            

        else:
            # ===== JOINT-SPACE SPLINE q0 -> qgoal =====
            qc, qcdot = goto(self.t, self.T, self.q0, self.qgoal)

        self.qc = qc
        self.qcdot = qcdot

        pc, Rc, Jv, Jw = self.chain.fkin(self.qc)
        vd = Jv @ qcdot
        wd = Jw @ qcdot
        pd = pc 
        Rd = Rc
        # # =======


        # if self.t > self.T:
        #     self.t = self.T 
        # (s, sdot) = goto(self.t, self.T, 0.0, 1.0)
        # direction = self.goal_p - self.p0
        # pd = self.p0 + s    * direction     # desired position in task space
        # vd =          sdot * direction      # desired linear velocity

        # Rd = self.goal_R                    # keep same orientation
        # wd = vzero()                        # no desired angular velocity

        

        # KINEMATICS
        # (_, _, Jv, Jw) = self.chain.fkin(self.qc)
        # J = np.vstack((Jv, Jw))
        # # get vr and wr from the vd + lamda(error concatenated) - equation in notes
        # linearv = vd + (0.1/self.dt) * self.ep
        # angularv = wd + (0.1/self.dt) * self.eR

        # xdot = np.concatenate((linearv, angularv))
        # self.qcdot = np.linalg.pinv(J) @ xdot
        # self.qc = self.qc + self.qcdot * self.dt
        # # run in again
        # (pc, Rc, Jv, Jw) = self.chain.fkin(self.qc)
        # self.ep = pd - pc
        # self.eR = eR(Rd, Rc)

        # qc = self.qc
        # qcdot = self.qcdot
        # self.L = 0.4
        # diagonal = np.diag([1/self.L, 1/self.L, 1/self.L, 1.0, 1.0, 1.0])
        # J_bar = diagonal @ J
        
        # ================================================
        header=Header(stamp=self.now.to_msg(), frame_id='panda_link0')
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
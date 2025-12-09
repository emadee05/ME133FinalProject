import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import numpy as np

from geometry_msgs.msg import PointStamped, TwistStamped, PoseStamped
from std_msgs.msg import Bool

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header
from visualization_msgs.msg import Marker

from .KinematicChain import KinematicChain
from .TrajectoryUtils import *
from .TransformHelpers import *
from geometry_msgs.msg import Point




class CombinedNode(Node):
    def __init__(self):
        super().__init__("combined")

        self.dt = 0.01
        

        # Physics parameters
        self.desired_launch_vel = None #nparray
        self.radius = 0.03
        self.ball_vz = 0.0
        self.ball_z = 1.0             # initial height
        self.restitution = 0.85
        self.min_bounce = 0.1
        self.initial_height = 1.0
        self.floor = 0.00

        # Ball XY position (currently hardcoded to paddle pos for testing)
        self.ball_x = 0.34
        self.ball_y = -0.1
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        self.frame = "panda_link0"

        # TF listener for paddle
        self.paddle_frame = "panda_paddle"

        # Paddle size (hardcoded for now)
        self.paddle_size = np.array([0.15, 0.2, 0.01])  # x_width, y_width, thickness


        ############## FROM PROJECT #################
        #################
        #################
        #################
        #################
        #################
        

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
        self.T = 3.0

        # === task-space goal
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)
        self.goal_p = self.p0 + np.array([0.1, 0.0, 0.1])  
        self.goal_R = self.R0       
        self.qdot_goal = np.zeros_like(self.q0)                 

        # Initialize the stored joint command position and task errors.
        self.qc = self.q0.copy()
        self.ep = vzero()
        self.eR = vzero()

        # Pick the convergence bandwidth.
        self.lam = 20
        self.pub = self.create_publisher(Marker, "ball_marker", 10)
        self.pos_pub = self.create_publisher(PointStamped, "ball_position", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "ball_velocity", 10)
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)

        self.goal_pos_sub = self.create_subscription(
            Point,
            '/goal_position', 
            self.goal_pos_callback,
            10
        )

        self.respawn_sub = self.create_subscription(
            Bool,
            "ball_respawn",
            self.respawn_callback,
            10
        )

        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        self.ball_vel = None   # np.array([vx, vy, vz])
        self.goal_pos = None # np.array([x,y,z])
        self.have_plan = False
        self.initial_height = 1.0      # must match ball node
        self.reset_eps_z = 0.02        # tolerance on height
        self.reset_eps_v = 0.05        # tolerance on velocity
        self.g = -3.0         # gravity
        self.v_paddle = None
        self.have_plan = False     # whether we've computed an intercept
        self.p_start = self.p0     # start pose for spline
        self.R_start = self.R0

        # Set up the timer to update at 100Hz, with (t=0) occuring in
        # the first update cycle (dt) from now.
        self.dt    = 0.01                       # 100Hz.
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time since 1970

        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))

        self.timer = self.create_timer(self.dt, self.update)
    

    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    def goal_pos_callback(self, msg: Point):
        self.goal_pos = np.array([
            msg.x,
            msg.y,
            msg.z
        ])

    def respawn_callback(self, msg):
        if msg.data:
            self.get_logger().info("Received respawn signal from goal node.")
            self.ball_z = self.initial_height
            self.ball_x, self.ball_y = self.random_xy()
            self.ball_vz = 0.0
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            self.have_plan = False
            



    def ik_to_joints(self, pd, Rd, qi = None, dt=0.01, c=0.5, max_iters=200, tol=1e-4):
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
            pos_err = pd - pc
            R_err_world = eR(Rd, Rc)
            paddle_err = Rc.T @ R_err_world
            paddle_err[2] = 0.0

            rot_err = Rc @ paddle_err
            
            err = np.hstack((pos_err, rot_err))
            J = np.vstack((Jv, Jw))
            if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < tol:
                return q, True
            # jac for pos, we care about the cartesian position
            dq = c*(np.linalg.pinv(J)@err)
            q = q + dq
        return q, False


    def R_from_normal(self, n_world: np.ndarray):
        n = np.asarray(n_world, dtype=float)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            raise ValueError("n_world must be non-zero")
        n = n / norm_n

        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        x = np.cross(a, n)
        x = x / np.linalg.norm(x)
        y = np.cross(n, x) 
        R = np.column_stack((x, y, n))
        return R


    def compute_launch_v_goal(self, p_launch, p_goal, T_flight):
        if T_flight<=0:
            raise ValueError("T_flight must be pos")
        g_vec = np.array([0.0, 0.0, self.g])
        return (p_goal - p_launch - 0.5*g_vec*(T_flight**2))/T_flight


    def random_xy(self):
        radius = 0.4
        theta = np.random.uniform(0, 2*np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y


    def respawn_ball(self):
        self.ball_z = self.initial_height
        self.ball_x, self.ball_y = self.random_xy()
        self.ball_vz = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        


    def update(self):
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt) 
        self.ball_vel = [self.ball_vx, self.ball_vy, self.ball_vz]
        self.ball_pos = [self.ball_x, self.ball_y, self.ball_z]
        z  = self.ball_z
        vz = self.ball_vz
        
        # Ball is back near spawn height and almost stationary -> treat as a new trial
        # if (abs(self.ball_z - self.initial_height) < self.reset_eps_z
        #         and np.linalg.norm(self.ball_vel) < self.reset_eps_v):
        #     if self.have_plan:
        #         self.get_logger().info("Ball respawned, clearing old plan.")
        #     self.have_plan = False
        if (not self.have_plan):
            z0  = self.ball_z
            vz0 = self.ball_vz

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

                        x_hit = self.ball_x
                        y_hit = self.ball_y
                        self.goal_p = np.array([x_hit, y_hit, z_hit])

                        if self.goal_pos is not None:
                            T_flight = 1
                            v_launch = self.compute_launch_v_goal(
                                self.goal_p, self.goal_pos, T_flight
                            )
                            self.v_paddle = v_launch - self.ball_vel 

                            # HOW TO BUILD THE GOAL ROTATION using v_paddle
                            # self.goal_R = self.R_from_normal(self.v_paddle)# TODOOOO
                            v_in  = np.array(self.ball_vel)
                            v_out = v_launch  # the desired outgoing velocity

                            n = v_in - v_out
                            n = n / np.linalg.norm(n)

                            self.goal_R = self.R_from_normal(n)

                            self.qgoal, self.reach = self.ik_to_joints(self.goal_p, self.goal_R) # also put desired orientation of tip

                            pc_goal, Rc_goal, Jv_goal, Jw_goal = self.chain.fkin(self.qgoal)
                            self.qdot_goal = np.linalg.pinv(Jv_goal) @ self.v_paddle
                            print("\n=== COMPUTED LAUNCH VELOCITY ===")
                            print("goal_p:", self.goal_p)
                            print("goal_pos:", self.goal_pos)
                            print("v_launch:", v_launch)
                            print("speed:", np.linalg.  norm(v_launch))
                            print("================================\n")

                            # ----- PASSING OVER THE NEW BALL LAUNCH VELOCITY -----
                            speed = np.linalg.norm(v_launch)
                            self.get_logger().info(
                                f"Launch vel command: {v_launch}, speed={speed:.3f}"
                            )
                            # ----------------------
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
            q0dot = np.zeros_like(self.q0)
            qc, qcdot = spline(self.t, self.T, self.q0, self.qgoal, q0dot, self.qdot_goal)



        self.qc = qc
        self.qcdot = qcdot

        pc, Rc, Jv, Jw = self.chain.fkin(self.qc)
        vd = Jv @ qcdot
        wd = Jw @ qcdot
        pd = pc 
        Rd = Rc


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
    



        ##################### FROM BALL MARKER
        #####################
        #####################
        #####################
        #####################
        #####################
        #####################
        #####################
        #####################
        self.ball_vz += self.g * self.dt
        self.ball_z += self.ball_vz * self.dt
        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt

        paddle_pos = pc
        paddle_rot = Rc
        # print("paddle_pos:", paddle_pos)
        if paddle_pos is not None and paddle_rot is not None:
           
            R = paddle_rot

            # Ball position in panda_link0
            ball_pos = np.array([self.ball_x, self.ball_y, self.ball_z])

            # Relative position in world frame
            rel_pos = ball_pos - paddle_pos    

            # Transform into paddle local frame
            rel_pos_local = R.T @ rel_pos

            half_size = self.paddle_size / 2

            in_xy = (
                -half_size[0] <= rel_pos_local[0] <= half_size[0] and
                -half_size[1] <= rel_pos_local[1] <= half_size[1]
            )

            touching_z = (rel_pos_local[2] - self.radius) <= half_size[2]

            if in_xy and touching_z:
               
                n = R[:, 2]
                if n[2] < 0:   # if pointing downward
                    n = -n 
                n = n / np.linalg.norm(n)

                # Ball velocity vector
                # --- Paddle linear velocity in world frame ---
                paddle_v = vd   # already computed by fkin/Jv above

                # --- Ball velocity vector ---
                v_ball = np.array([self.ball_vx, self.ball_vy, self.ball_vz])

                # --- Relative velocity (ball w.r.t moving paddle) ---
                v_rel = v_ball - paddle_v

                v_rel = v_ball - paddle_v


                # --- Split into normal and tangential components ---
                v_rel_n = np.dot(v_rel, n) * n
                v_rel_t = v_rel - v_rel_n

                # --- Bounce (reflect normal component with restitution) ---
                v_rel_n_after = -self.restitution * v_rel_n

                # --- New relative velocity ---
                v_rel_after = v_rel_t + v_rel_n_after

                # --- Convert back to world frame ---
                v_after = v_rel_after + paddle_v

                # --- Update ball velocity ---
                v_after = (np.linalg.norm(v_ball)) * n
                self.ball_vx = float(v_after[0])
                self.ball_vy = float(v_after[1])
                self.ball_vz = float(v_after[2])

                self.get_logger().info(
                    f"CONTACT PHYSICS (with paddle motion): v = ({self.ball_vx:.2f}, {self.ball_vy:.2f}, {self.ball_vz:.2f})"
                )

                # Respawn if bounce becomes too small
                if abs(self.ball_vz) < self.min_bounce and np.linalg.norm(v_after) < 0.2:
                    self.get_logger().info("Bounce too small -> respawn")
                    self.ball_z = self.initial_height
                    self.ball_x, self.ball_y = self.random_xy()
                    self.ball_vx = self.ball_vy = self.ball_vz = 0.0


        # --- Floor collision ---
        if self.ball_z - self.radius <= self.floor:
            self.ball_z = self.floor + self.radius
            self.ball_vz = -self.ball_vz * self.restitution
            if abs(self.ball_vz) < self.min_bounce:
                self.ball_z = self.initial_height
                self.ball_x, self.ball_y = self.random_xy()
                self.ball_vz = 0.0
                self.ball_vx = 0.0
                self.ball_vy = 0.0

        # Ball marker 
        ball_marker = Marker()
        ball_marker.header.frame_id = self.frame
        ball_marker.header.stamp = self.get_clock().now().to_msg()
        ball_marker.ns = "ball"
        ball_marker.id = 0
        ball_marker.type = Marker.SPHERE
        ball_marker.action = Marker.ADD
        ball_marker.pose.position.x = self.ball_x
        ball_marker.pose.position.y = self.ball_y
        ball_marker.pose.position.z = self.ball_z
        ball_marker.pose.orientation.w = 1.0
        ball_marker.scale.x = self.radius * 2
        ball_marker.scale.y = self.radius * 2
        ball_marker.scale.z = self.radius * 2
        ball_marker.color = ColorRGBA(r=1.0, g=0.4, b=0.1, a=1.0)
        ball_marker.lifetime.sec = 0
        self.pub.publish(ball_marker)

        pos_msg = PointStamped()
        pos_msg.header.frame_id = self.frame
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.point.x = self.ball_x
        pos_msg.point.y = self.ball_y
        pos_msg.point.z = self.ball_z
        self.pos_pub.publish(pos_msg)

        # --- Publish ball velocity ---
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = self.frame
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.twist.linear.x = self.ball_vx
        vel_msg.twist.linear.y = self.ball_vy
        vel_msg.twist.linear.z = self.ball_vz
        self.vel_pub.publish(vel_msg)

        


def main(args=None):
    rclpy.init(args=args)
    node = CombinedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping ball node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
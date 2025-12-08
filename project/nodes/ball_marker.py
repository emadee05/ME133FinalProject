import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import numpy as np
import tf2_ros
from tf_transformations import quaternion_from_matrix
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import Bool



class BouncingBall(Node):
    def __init__(self):
        super().__init__("bouncing_ball_marker")

        self.pub = self.create_publisher(Marker, "ball_marker", 10)
        self.pos_pub = self.create_publisher(PointStamped, "ball_position", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "ball_velocity", 10)
        self.desired_launch_vel = None
        self.launch_sub = self.create_subscription(
            TwistStamped,
            "ball_launch_vel",
            self.launch_callback,
            10
        )

        self.respawn_sub = self.create_subscription(
            Bool,
            "ball_respawn",
            self.respawn_callback,
            10
        )

        # 100Hz
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.update)

        # Physics parameters
        self.desired_launch_vel = None #nparray
        self.g = -9.8          # gravity
        self.radius = 0.03
        self.vz = 0.0
        self.z = 1.0             # initial height
        self.restitution = 0.85
        self.min_bounce = 0.1
        self.initial_height = 1.0
        self.floor = 0.00

        # Ball XY position (currently hardcoded to paddle pos for testing)
        self.x = 0.34
        self.y = -0.1
        self.vx = 0.0
        self.vy = 0.0

        self.frame = "panda_link0"

        # TF listener for paddle
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.paddle_frame = "panda_paddle"

        # Paddle size (hardcoded for now)
        self.paddle_size = np.array([0.15, 0.2, 0.01])  # x_width, y_width, thickness


    def launch_callback(self, msg: TwistStamped):
        self.desired_launch_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ], dtype=float)

    def random_xy(self):
        radius = 0.4
        theta = np.random.uniform(0, 2*np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y


    def respawn_callback(self, msg):
        if msg.data:
            self.get_logger().info("Received respawn signal from goal node.")
            self.z = self.initial_height
            self.x, self.y = self.random_xy()
            self.vz = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.desired_launch_vel = None

    def get_paddle_transform(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'panda_link0',   # target frame
                self.paddle_frame,
                rclpy.time.Time(),  # latest
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            rot = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            return pos, rot
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None, None

    def launch_callback(self, msg: TwistStamped):
        self.desired_launch_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ], dtype=float)
        
    def update(self):
        self.vz += self.g * self.dt
        self.z += self.vz * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        paddle_pos, paddle_rot = self.get_paddle_transform()
        if paddle_pos is not None:
            #offset from joint to paddle center
            paddle_offset = np.array([0.0, 0.0, 0.0])
            paddle_center = paddle_pos + paddle_offset

            # quaternion to rotation matrix
            qx, qy, qz, qw = paddle_rot
            R = quaternion_to_matrix([qx, qy, qz, qw])[:3, :3]

            # Collision detection in paddle frame
            rel_pos = np.array([self.x, self.y, self.z]) - paddle_center
            rel_pos_local = R.T @ rel_pos  # transform to paddle local frame

            half_size = self.paddle_size / 2
            in_xy = (-half_size[0] <= rel_pos_local[0] <= half_size[0]) and (-half_size[1] <= rel_pos_local[1] <= half_size[1])
            touching_z = rel_pos_local[2] - self.radius <= half_size[2]

            if in_xy and touching_z:
                # --- Correct ball position so it sits exactly on paddle surface ---
                rel_pos_local[2] = half_size[2] + self.radius
                new_world_pos = R @ rel_pos_local + paddle_center
                self.x, self.y, self.z = new_world_pos

                # --- Infinite-mass elastic collision physics ---

                # Paddle normal in world frame (paddle's +Z axis)
                n = R[:, 2]
                n = n / np.linalg.norm(n)

                # Ball velocity vector
                v = np.array([self.vx, self.vy, self.vz])

                # Decompose velocity into normal/tangential components
                v_n = np.dot(v, n) * n     # normal component
                v_t = v - v_n              # tangential component

                # Reflect normal component using restitution
                v_n_after = -self.restitution * v_n

                # Final post-collision velocity
                v_after = v_t + v_n_after

                self.vx = float(v_after[0])
                self.vy = float(v_after[1])
                self.vz = float(v_after[2])

                self.get_logger().info(
                    f"CONTACT PHYSICS: new v = ({self.vx:.2f}, {self.vy:.2f}, {self.vz:.2f})"
                )

                # Respawn if bounce becomes too small
                if abs(self.vz) < self.min_bounce and np.linalg.norm(v_after) < 0.2:
                    self.get_logger().info("Bounce too small -> respawn")
                    self.z = self.initial_height
                    self.x, self.y = self.random_xy()
                    self.vx = self.vy = self.vz = 0.0


        # --- Floor collision ---
        if self.z - self.radius <= self.floor:
            self.z = self.floor + self.radius
            self.vz = -self.vz * self.restitution
            if abs(self.vz) < self.min_bounce:
                self.z = self.initial_height
                self.x, self.y = self.random_xy()
                self.vz = 0.0
                self.vx = 0.0
                self.vy = 0.0

        # Ball marker 
        ball_marker = Marker()
        ball_marker.header.frame_id = self.frame
        ball_marker.header.stamp = self.get_clock().now().to_msg()
        ball_marker.ns = "ball"
        ball_marker.id = 0
        ball_marker.type = Marker.SPHERE
        ball_marker.action = Marker.ADD
        ball_marker.pose.position.x = self.x
        ball_marker.pose.position.y = self.y
        ball_marker.pose.position.z = self.z
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
        pos_msg.point.x = self.x
        pos_msg.point.y = self.y
        pos_msg.point.z = self.z
        self.pos_pub.publish(pos_msg)

        # --- Publish ball velocity ---
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = self.frame
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.twist.linear.x = self.vx
        vel_msg.twist.linear.y = self.vy
        vel_msg.twist.linear.z = self.vz
        self.vel_pub.publish(vel_msg)


def quaternion_to_matrix(q):
    """Convert quaternion [x,y,z,w] to 4x4 homogeneous rotation matrix"""
    x, y, z, w = q
    mat = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w,     2*x*z+2*y*w,     0],
        [2*x*y+2*z*w,     1-2*x**2-2*z**2, 2*y*z-2*x*w,     0],
        [2*x*z-2*y*w,     2*y*z+2*x*w,     1-2*x**2-2*y**2, 0],
        [0, 0, 0, 1]
    ])
    return mat


def main(args=None):
    rclpy.init(args=args)
    node = BouncingBall()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping ball node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

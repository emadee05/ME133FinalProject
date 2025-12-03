import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import numpy as np
import tf2_ros

class BouncingBall(Node):
    def __init__(self):
        super().__init__("bouncing_ball_marker")

        # Publisher for RViz markers
        self.pub = self.create_publisher(Marker, "ball_marker", 10)

        # Timer 100Hz
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.update)

        # Physics parameters
        self.g = -5           # gravity
        self.radius = 0.03
        self.vz = 0.0
        self.z = 1.0             # initial height
        self.restitution = 0.85
        self.min_bounce = 0.1
        self.initial_height = 1.0
        self.floor = 0.00        # floor z

        # Ball XY position
        self.x = 0.34
        self.y = -0.1

        # Appearance
        self.frame = "panda_link0"

        # TF listener for paddle
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.paddle_frame = "panda_paddle"  # frame broadcast by hovering arm

        # Paddle size (hardcoded for now)
        self.paddle_size = np.array([0.15, 0.2, 0.01])  # x_width, y_width, thickness

        self.get_logger().info("Bouncing ball marker running.")

    def get_paddle_position(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'panda_link0',   # target frame
                self.paddle_frame,
                rclpy.time.Time(),  # latest
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def update(self):
        # --- Ball physics ---
        self.vz += self.g * self.dt
        self.z += self.vz * self.dt

        # --- Get paddle position and apply offset ---
        paddle_pos = self.get_paddle_position()
        paddle_offset = np.array([-0.13, -0.15, 0.0])
        if paddle_pos is not None:
            paddle_pos_corrected = paddle_pos + paddle_offset

            # Publish debug marker for paddle
        #     paddle_marker = Marker()
        #     paddle_marker.header.frame_id = "panda_link0"
        #     paddle_marker.header.stamp = self.get_clock().now().to_msg()
        #     paddle_marker.ns = "paddle_debug"
        #     paddle_marker.id = 1
        #     paddle_marker.type = Marker.CUBE
        #     paddle_marker.action = Marker.ADD
        #     paddle_marker.pose.position.x = paddle_pos_corrected[0]
        #     paddle_marker.pose.position.y = paddle_pos_corrected[1]
        #     paddle_marker.pose.position.z = paddle_pos_corrected[2]
        #     paddle_marker.pose.orientation.w = 1.0
        #     paddle_marker.scale.x = self.paddle_size[0]
        #     paddle_marker.scale.y = self.paddle_size[1]
        #     paddle_marker.scale.z = self.paddle_size[2]
        #     paddle_marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.5)  # semi-transparent
        #     paddle_marker.lifetime.sec = 0
        #     self.pub.publish(paddle_marker)
        # else:
        #     paddle_pos_corrected = None

        # --- Collision detection ---
            bounced = False
            if paddle_pos_corrected is not None:
                paddle_min = paddle_pos_corrected - self.paddle_size / 2
                paddle_max = paddle_pos_corrected + self.paddle_size / 2

                in_xy = (paddle_min[0] <= self.x <= paddle_max[0]) and (paddle_min[1] <= self.y <= paddle_max[1])
                touching_z = (self.z - self.radius) <= paddle_max[2]

                if in_xy and touching_z:
                    self.z = paddle_max[2] + self.radius
                    self.vz = -self.vz * self.restitution
                    bounced = True
                    self.get_logger().info(f"CONTACT! Ball hit the paddle at z={self.z:.3f}")

                    # --- Respawn if too small ---
                    if abs(self.vz) < self.min_bounce:
                        self.get_logger().info("Velocity too small on paddle, respawning ball.")
                        self.z = self.initial_height
                        self.vz = 0.0

        # --- Ball marker ---
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

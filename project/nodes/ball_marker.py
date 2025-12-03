import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class BouncingBall(Node):
    def __init__(self):
        super().__init__("bouncing_ball_marker")

        # Publisher for RViz markers
        self.pub = self.create_publisher(Marker, "ball_marker", 10)

        # Timer 100Hz
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.update)

        # Physics parameters
        self.g = -9.81                     # gravity
        self.z = 1.0                       # initial height
        self.vz = 0.0                      # initial velocity
        self.restitution = 0.85            # bounce energy retention
        self.ground = 0.05                 # height of the ball center at "ground"

        # Appearance
        self.radius = 0.07
        self.frame = "panda_link0"        # parent frame

        self.get_logger().info("Bouncing ball marker running.")

    def update(self):
        # Physics integration
        self.vz += self.g * self.dt
        self.z += self.vz * self.dt

        # Bounce
        if self.z <= self.ground:
            self.z = self.ground
            self.vz = -self.vz * self.restitution

        # Publish marker
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ball"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = self.z

        # Orientation
        marker.pose.orientation.w = 1.0

        # Size
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.radius * 2

        # Color
        marker.color = ColorRGBA(r=1.0, g=0.4, b=0.1, a=1.0)

        marker.lifetime.sec = 0  
        self.pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BouncingBall()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool

from std_msgs.msg import ColorRGBA
import numpy as np


class GoalMarker(Node):
    def __init__(self):
        super().__init__("goal_marker")

        self.marker_pub = self.create_publisher(Marker, "goal_marker", 10)
        self.goal_pub = self.create_publisher(Point, "goal_position", 10)
        self.respawn_pub = self.create_publisher(Bool, "ball_respawn", 10)


        self.ball_sub = self.create_subscription(
            PointStamped,
            "ball_position",
            self.ball_callback,
            10
        )

        self.ball_x = None
        self.ball_y = None
        self.ball_z = None

        self.goal_threshold = 0.3

        self.spawn_radius = 2.0
        self.timer = self.create_timer(0.1, self.update)
        self.respawn_goal()

    def ball_callback(self, msg):
        self.ball_x = msg.point.x
        self.ball_y = msg.point.y
        self.ball_z = msg.point.z

    def respawn_goal(self):
        angle = np.random.uniform(0, 2 * np.pi)
        self.goal_x = self.spawn_radius * np.cos(angle)
        self.goal_y = self.spawn_radius * np.sin(angle)
        self.goal_z = 0.1

        self.get_logger().info(
            f"New goal at x={self.goal_x:.3f}, y={self.goal_y:.3f}, z={self.goal_z:.3f}"
        )

    def update(self):
        self.check_hit()
        self.publish_goal()


    def check_hit(self):
        if self.ball_x is None:
            return 

        dist = np.sqrt(
            (self.ball_x - self.goal_x)**2 +
            (self.ball_y - self.goal_y)**2 +
            (self.ball_z - self.goal_z)**2
        )

        if dist < self.goal_threshold:
            self.get_logger().info(
                f"Ball reached goal! dist={dist:.3f}. Respawning goal."
            )
            self.respawn_pub.publish(Bool(data=True))
            self.respawn_goal()

    def publish_goal(self):
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = self.goal_z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6

        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
        self.marker_pub.publish(marker)

        pt = Point()
        pt.x = self.goal_x
        pt.y = self.goal_y
        pt.z = self.goal_z
        self.goal_pub.publish(pt)


def main(args=None):
    rclpy.init(args=args)
    node = GoalMarker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

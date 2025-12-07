import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class GoalMarker(Node):
    def __init__(self):
        super().__init__('goal_marker_node')

        # Publisher for the goal marker
        self.marker_pub = self.create_publisher(Marker, 'goal_marker', 10)

        # Subscriber to the ball position
        self.ball_sub = self.create_subscription(
            Point,
            '/ball_position',
            self.ball_callback,
            10
        )

        # Initialize goal marker
        self.goal_marker = Marker()
        self.goal_marker.header.frame_id = "world"
        self.goal_marker.ns = "goal"
        self.goal_marker.id = 0
        self.goal_marker.type = Marker.SPHERE
        self.goal_marker.action = Marker.ADD
        self.goal_marker.scale.x = 0.1
        self.goal_marker.scale.y = 0.1
        self.goal_marker.scale.z = 0.1
        self.goal_marker.color.r = 1.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 0.0
        self.goal_marker.color.a = 1.0

        # Spawn the first goal
        self.respawn_goal()

        # Distance threshold for “hitting” the goal
        self.hit_threshold = 0.15

    def respawn_goal(self):
        # Random position inside some bounds (adjust as needed)
        self.goal_marker.pose.position.x = np.random.uniform(-1.0, 1.0)
        self.goal_marker.pose.position.y = np.random.uniform(-1.0, 1.0)
        self.goal_marker.pose.position.z = np.random.uniform(0.0, 0.5)
        self.goal_marker.header.stamp = self.get_clock().now().to_msg()
        self.goal_marker.action = Marker.ADD
        self.marker_pub.publish(self.goal_marker)
        self.get_logger().info(
            f"Goal spawned at ({self.goal_marker.pose.position.x:.2f}, "
            f"{self.goal_marker.pose.position.y:.2f}, "
            f"{self.goal_marker.pose.position.z:.2f})"
        )

    def ball_callback(self, msg):
        # Compute distance to current goal
        dx = msg.x - self.goal_marker.pose.position.x
        dy = msg.y - self.goal_marker.pose.position.y
        dz = msg.z - self.goal_marker.pose.position.z
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < self.hit_threshold:
            self.get_logger().info("Goal hit! Respawning...")
            self.respawn_goal()


def main(args=None):
    rclpy.init(args=args)
    node = GoalMarker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

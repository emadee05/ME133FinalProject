import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class GoalMarker(Node):
    def __init__(self):
        super().__init__("goal_marker")

        # Publisher for visualization in RViz
        self.marker_pub = self.create_publisher(Marker, "goal_marker", 10)

        # Publisher for controllers to read goal XYZ
        self.goal_pub = self.create_publisher(Point, "goal_position", 10)

        # Timer to periodically re-publish the static marker
        self.timer = self.create_timer(0.1, self.publish_goal)

        # Define static goal location (change as needed)
        self.goal_x = 2.0
        self.goal_y = 2.0
        self.goal_z = 0.0

    def publish_goal(self):

        # ---- publish the marker ----
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

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)

        self.marker_pub.publish(marker)

        # ---- publish the point for control nodes ----
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

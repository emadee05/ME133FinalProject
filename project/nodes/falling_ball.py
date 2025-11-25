import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import random

class FallingBall(Node):
    def __init__(self):
        super().__init__('falling_ball')

        self.tf_broadcaster = TransformBroadcaster(self)

        # Ball settings
        self.start_z = 1.5  # initial height
        self.reset_on_ground = True
        self.dt = 0.02      # 50 Hz update
        self.v = -0.2       # falling speed m/s

        # Randomize x/y for each drop
        self.x = random.uniform(-0.2, 0.2)
        self.y = random.uniform(-0.2, 0.2)
        self.z = self.start_z

        self.timer = self.create_timer(self.dt, self.update)

    def update(self):
        self.z += self.v * self.dt

        # If hit ground
        if self.z <= 0.0:
            if self.reset_on_ground:
                # Reset position for another fall
                self.z = self.start_z
                self.x = random.uniform(-0.2, 0.2)
                self.y = random.uniform(-0.2, 0.2)
                self.get_logger().info(f'Ball reset to ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})')
            else:
                self.destroy_timer(self.timer)
                self.get_logger().info('Ball hit the ground, stopping TF')
                return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "panda_link0"
        t.child_frame_id = "ball_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = FallingBall()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

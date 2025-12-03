import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import random


class FallingBall(Node):
    def __init__(self):
        super().__init__('falling_ball')

        self.tf_broadcaster = TransformBroadcaster(self)

        # Ball physics
        self.start_z = 1.5      # starting height
        self.dt = 0.02          # time step (50 Hz)
        self.g = -9.81          # gravity
        self.vz = 0.0           # vertical velocity

        # Coefficient of restitution (0 = dead, 1 = perfect bounce)
        self.restitution = 0.7

        # Ball position
        self.x = random.uniform(-0.2, 0.2)
        self.y = random.uniform(-0.2, 0.2)
        self.z = self.start_z

        self.timer = self.create_timer(self.dt, self.update)


    def update(self):

        # Apply gravity
        self.vz += self.g * self.dt
        self.z += self.vz * self.dt

        # Ground hit
        if self.z <= 0.0:
            self.z = 0.0

            # Reverse velocity with damping
            self.vz = -self.vz * self.restitution
            self.get_logger().info(f'Bounce: vz={self.vz:.2f}')

            # If too slow, reset to top
            if abs(self.vz) < 0.3:
                self.z = self.start_z
                self.vz = 0.0
                self.x = random.uniform(-0.2, 0.2)
                self.y = random.uniform(-0.2, 0.2)
                self.get_logger().info(
                    f'Reset ball to ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})'
                )

        # Publish TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "panda_link0"
        t.child_frame_id = "ball_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
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

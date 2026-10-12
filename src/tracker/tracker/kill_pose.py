#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped

class TargetPointBroadcaster(Node):
    def __init__(self):
        super().__init__('target_point_broadcaster')

        self.declare_parameter('parent_frame', 'enemy_bot/base_link')
        self.declare_parameter('child_frame', 'enemy_bot/target_point')
        
        # TUNING POINT: How far directly behind the enemy the target point sits.
        # If your bot crashes into the back of the enemy, increase this. If it stops too early, decrease it.
        self.declare_parameter('distance_behind', 1.0)

        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.distance_behind = self.get_parameter('distance_behind').value

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(1.0 / 30.0, self.broadcast_target_frame)

    def broadcast_target_frame(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        # orientation of the tf
        t.transform.translation.x = -float(self.distance_behind)
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = TargetPointBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
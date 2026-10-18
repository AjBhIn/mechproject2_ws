#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped

# --- GLOBAL TUNING VARIABLES ---
# Reduced from 1.0m to 0.45m to keep the target clear of massive 0.35m costmap inflation zones.
DEFAULT_DISTANCE_BEHIND = 0.45 
# TF broadcast rate (~30Hz) for smooth, high-fidelity target tracking
BROADCAST_RATE_SEC = 1.0 / 30.0 

class TargetPointBroadcaster(Node):
    """
    Broadcasts the theoretical 'Carrot' or capture target point relative to the enemy robot.
    Places a continuous TF frame exactly behind the enemy's rear axis, optimized to 
    remain within strike distance but outside of lethal inflation ranges.
    """
    def __init__(self):
        super().__init__('target_point_broadcaster')

        # --- Dynamic Namespace Configuration ---
        raw_ns = self.get_namespace().strip('/')
        default_enemy_ns = raw_ns if raw_ns else 'enemy_bot'

        self.declare_parameter('enemy', default_enemy_ns)
        self.enemy_ns = self.get_parameter('enemy').value.strip('/')

        default_parent = f'{self.enemy_ns}/base_link' if self.enemy_ns else 'base_link'
        default_child = f'{self.enemy_ns}/target_point' if self.enemy_ns else 'target_point'

        self.declare_parameter('parent_frame', default_parent)
        self.declare_parameter('child_frame', default_child)
        self.declare_parameter('distance_behind', DEFAULT_DISTANCE_BEHIND)

        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.distance_behind = self.get_parameter('distance_behind').value

        # --- TF2 Broadcaster & Timer ---
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(BROADCAST_RATE_SEC, self.broadcast_target_frame)
        self.get_logger().info(f"[KILL POSE] Initialized. Broadcasting target {self.distance_behind}m behind enemy.")

    def broadcast_target_frame(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        # Position the target point directly behind the enemy on its local X-axis
        t.transform.translation.x = -float(self.distance_behind)
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # Align orientation perfectly parallel with the enemy's heading
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
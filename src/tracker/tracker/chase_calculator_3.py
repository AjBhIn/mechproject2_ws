#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ChaserCalculator(Node):
    def __init__(self):
        super().__init__('chaser_calculator')

        self.map_frame = 'map'
        self.pursuer_frame = 'our_bot/base_link'
        self.carrot_frame = 'enemy_bot/target_point'
        self.broadcast_frame = 'chase_nav_target'

        # TUNING POINT: Intermediate goal projection horizon
        self.short_goal_dist = 0.8
        self.capture_orientation_dist = 0.35

        self.current_state = 'CHASING'
        self.costmap = None
        self.last_valid_yaw = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.create_subscription(OccupancyGrid, '/our_bot/global_costmap/costmap', self.costmap_cb, 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Chaser Calculator initialized. Safe linear retraction active.")

    def state_cb(self, msg):
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg

    def is_pose_safe(self, x: float, y: float, max_cost: int = 50) -> bool:
        if self.costmap is None: return True
        info = self.costmap.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)

        # MAP BOUNDARY GUARD: Enforce a 15-cell margin (0.75m) from map edges
        MARGIN = 15
        if gx < MARGIN or gx >= (info.width - MARGIN) or \
           gy < MARGIN or gy >= (info.height - MARGIN):
            return False

        cost = self.costmap.data[gy * info.width + gx]
        return 0 <= cost < max_cost

    def project_to_free_space(self, start_x: float, start_y: float, target_x: float, target_y: float):
        # Linear Retraction: Pull target back towards pursuer along line-of-sight vector until safe
        if self.is_pose_safe(target_x, target_y, max_cost=50):
            return target_x, target_y

        dist = math.hypot(target_x - start_x, target_y - start_y)
        if dist < 0.05: return start_x, start_y

        for i in range(1, 20):
            alpha = 1.0 - (i / 20.0)
            px = start_x + alpha * (target_x - start_x)
            py = start_y + alpha * (target_y - start_y)
            if self.is_pose_safe(px, py, max_cost=50):
                return px, py

        return start_x, start_y

    def control_loop(self):
        if self.current_state != 'CHASING':
            return

        try:
            t_our = self.tf_buffer.lookup_transform(self.map_frame, self.pursuer_frame, rclpy.time.Time())
            t_carrot = self.tf_buffer.lookup_transform(self.map_frame, self.carrot_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return

        p_x, p_y = t_our.transform.translation.x, t_our.transform.translation.y
        c_x, c_y = t_carrot.transform.translation.x, t_carrot.transform.translation.y
        c_rot = t_carrot.transform.rotation

        dx = c_x - p_x
        dy = c_y - p_y
        dist = math.hypot(dx, dy)

        # Dynamic goal lead distance
        lead_dist = min(self.short_goal_dist, max(0.4, dist * 0.7))

        if dist > lead_dist:
            ratio = lead_dist / dist
            target_x = p_x + dx * ratio
            target_y = p_y + dy * ratio
        else:
            target_x = c_x
            target_y = c_y

        # Prevent 180-degree move_yaw flips on micro-overshoots
        if dist > 0.15:
            self.last_valid_yaw = math.atan2(dy, dx)

        if dist <= self.capture_orientation_dist:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        safe_x, safe_y = self.project_to_free_space(p_x, p_y, target_x, target_y)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.broadcast_frame
        t.transform.translation.x = safe_x
        t.transform.translation.y = safe_y
        t.transform.rotation.z = rot_z
        t.transform.rotation.w = rot_w

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ChaserCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()